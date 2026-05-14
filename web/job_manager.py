import os
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Jobs older than this are purged from memory (output files kept on disk)
_JOB_MAX_AGE_HOURS = 48
# Hard timeout: kill pipeline if it runs longer than this
_JOB_TIMEOUT_SECONDS = 7200  # 2 hours


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
