import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Jobs older than this are purged from memory (output files kept on disk)
_JOB_MAX_AGE_HOURS = 48
# Hard timeout: kill pipeline if it runs longer than this.
# Defaults to 12 h — a 287M-point E57 read + ML segment can comfortably
# take several hours on a single GPU. Overridable via env var for users
# who want to tighten/relax it without code changes.
_JOB_TIMEOUT_SECONDS = int(os.environ.get("CLOUD2BIM_JOB_TIMEOUT_SECONDS", str(12 * 60 * 60)))

# Cap the number of log lines kept in memory / restored from disk on
# rehydration. A long ML run can emit tens of thousands of lines and we
# only need a tail for "what went wrong" — anything older isn't useful.
_LOG_TAIL_LIMIT = 5000


class JobManager:
    def __init__(self, jobs_dir: Path):
        self.jobs_dir = jobs_dir
        self._jobs: dict = {}
        self._lock = threading.Lock()

    def create_job(self, job_id: str, input_path: str, mode: str = "full") -> dict:
        self._evict_old_jobs()
        with self._lock:
            job = {
                "job_id": job_id,
                "status": "pending",
                "mode": mode,
                "input_path": input_path,
                "log_lines": [],
                "current_stage": None,
                "created_at": datetime.now().isoformat(),
                "finished_at": None,
            }
            self._jobs[job_id] = job
            return dict(job)

    def ensure_job(
        self, job_id: str, input_path: str, mode: str = "stepwise"
    ) -> dict:
        """Idempotent create — returns existing entry if any, else creates one.

        Used to lazy-register a job in memory when the user kicks off a
        stage on a job whose JobManager entry was lost (server restart
        before rehydration, or a job that pre-dates persistence). The
        new entry's log_lines is seeded from the on-disk tail so re-
        attaching to the SSE log shows the failure that led to the retry.
        """
        with self._lock:
            existing = self._jobs.get(job_id)
            if existing is not None:
                return dict(existing)
            job = {
                "job_id": job_id,
                "status": "failed",
                "mode": mode,
                "input_path": input_path,
                "log_lines": _load_log_tail(self.jobs_dir / job_id),
                "current_stage": None,
                "created_at": datetime.now().isoformat(),
                "finished_at": datetime.now().isoformat(),
            }
            self._jobs[job_id] = job
            return dict(job)

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list_jobs(self) -> list:
        with self._lock:
            return [
                {k: v for k, v in j.items() if k != "log_lines"}
                for j in self._jobs.values()
            ]

    def rehydrate_from_disk(self) -> int:
        """Register every job_dir on disk that isn't already in memory.

        Run once at server startup. Any job_dir with a config.yaml but no
        live in-memory entry is treated as `failed` (the subprocess that
        was running before the restart is gone). Completed jobs (those
        whose output.ifc exists) are recorded with status='completed' so
        the UI can still hit /api/jobs/{id}/* endpoints against them.
        Returns the number of jobs rehydrated.
        """
        if not self.jobs_dir.exists():
            return 0
        count = 0
        for job_dir in self.jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue
            job_id = job_dir.name
            if job_id in self._jobs:
                continue
            config_path = job_dir / "config.yaml"
            if not config_path.exists():
                continue
            info = _read_json(job_dir / "job_info.json")
            state = _read_json(job_dir / "state.json")
            output_ifc = job_dir / "output.ifc"
            status = "completed" if output_ifc.exists() else "failed"
            mode = "stepwise"  # stages are resumable from any mode
            created_at = info.get("created_at") or datetime.now().isoformat()
            # Last completed stage timestamp is the closest thing we have
            # to a "finished_at" for a job whose process is long gone.
            finished_at = max(state.values()) if state else created_at
            with self._lock:
                self._jobs[job_id] = {
                    "job_id": job_id,
                    "status": status,
                    "mode": mode,
                    "input_path": "",
                    "log_lines": _load_log_tail(job_dir),
                    "current_stage": None,
                    "created_at": created_at,
                    "finished_at": finished_at,
                    # Distinguish "process exited non-zero" (true failure)
                    # from "the server lost the in-memory entry" (looks
                    # the same on disk but means something different to
                    # the user). Cleared the next time a subprocess
                    # actually runs against this job.
                    "rehydrated": status != "completed",
                }
            count += 1
        return count

    def run_job(self, job_id: str, config_path: str, preprocess_fn=None):
        """Blocking — run the full pipeline. Used for ``mode='full'`` jobs."""
        project_root = Path(__file__).parent.parent
        self._set_status(job_id, "running")
        process = None
        try:
            if preprocess_fn is not None:
                self._append_log(job_id, "[INFO] Förbereder fil...")
                preprocess_fn(lambda msg: self._append_log(job_id, msg))
                self._append_log(job_id, "[INFO] Filkonvertering klar.")

            process = subprocess.Popen(
                [sys.executable, "-m", "cloud2bim", "run", config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(project_root),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            status = self._stream_subprocess(job_id, process)
        except Exception as exc:
            self._append_log(job_id, f"[ERROR] {exc}")
            if process is not None and process.poll() is None:
                process.kill()
            status = "failed"

        self._set_status(job_id, status)

    def run_stages_async(self, job_id: str, config_path: str, stages: list[str]):
        """Run one or more named stages in sequence (used by wizard mode)."""
        project_root = Path(__file__).parent.parent
        self._set_status(job_id, "running")

        last_status = "completed"
        for stage in stages:
            self._set_current_stage(job_id, stage)
            self._append_log(job_id, f"[INFO] ── stage: {stage} ──")
            process = None
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "cloud2bim", "step", config_path, stage],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(project_root),
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                stage_status = self._stream_subprocess(job_id, process)
            except Exception as exc:
                self._append_log(job_id, f"[ERROR] {exc}")
                if process is not None and process.poll() is None:
                    process.kill()
                stage_status = "failed"

            if stage_status != "completed":
                last_status = "failed"
                break

        self._set_current_stage(job_id, None)
        self._set_status(job_id, last_status if last_status == "completed" else "failed")

    def _stream_subprocess(self, job_id: str, process: subprocess.Popen) -> str:
        """Drain a subprocess into the job log and return a job status."""
        _timeout_hit = [False]

        def _kill_after_timeout():
            if process.poll() is None:
                _timeout_hit[0] = True
                process.kill()

        timer = threading.Timer(_JOB_TIMEOUT_SECONDS, _kill_after_timeout)
        timer.daemon = True
        timer.start()
        try:
            for line in process.stdout:
                self._append_log(job_id, line.rstrip())
            process.wait()
        finally:
            timer.cancel()

        if _timeout_hit[0]:
            self._append_log(
                job_id,
                "[ERROR] Jobb avbröts — överskred tidsgränsen (%d min)."
                % (_JOB_TIMEOUT_SECONDS // 60),
            )
            return "failed"
        return "completed" if process.returncode == 0 else "failed"

    # ── internal helpers ────────────────────────────────────────────────────

    def _set_status(self, job_id: str, status: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = status
                if status == "running":
                    # A real subprocess is now running against this job —
                    # whatever happens next is an authoritative outcome,
                    # not an inherited rehydrated state.
                    self._jobs[job_id]["rehydrated"] = False
                if status in ("completed", "failed"):
                    self._jobs[job_id]["finished_at"] = datetime.now().isoformat()

    def _set_current_stage(self, job_id: str, stage: Optional[str]):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["current_stage"] = stage

    def _append_log(self, job_id: str, line: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["log_lines"].append(line)
        # Persist to disk so rehydrated jobs can still show their tail.
        # Best-effort: a failed write must not break the pipeline.
        try:
            log_path = self.jobs_dir / job_id / "log.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass

    def _evict_old_jobs(self):
        """Remove completed/failed jobs older than _JOB_MAX_AGE_HOURS from memory."""
        cutoff = datetime.now() - timedelta(hours=_JOB_MAX_AGE_HOURS)
        with self._lock:
            to_remove = [
                jid for jid, job in self._jobs.items()
                if job["status"] in ("completed", "failed")
                and job.get("finished_at")
                and datetime.fromisoformat(job["finished_at"]) < cutoff
            ]
            for jid in to_remove:
                del self._jobs[jid]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_log_tail(job_dir: Path) -> list:
    """Read the last _LOG_TAIL_LIMIT lines of log.txt, or [] if missing."""
    path = job_dir / "log.txt"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        return lines[-_LOG_TAIL_LIMIT:]
    except Exception:
        return []
