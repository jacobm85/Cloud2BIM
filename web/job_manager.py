import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


class JobManager:
    def __init__(self, jobs_dir: Path):
        self.jobs_dir = jobs_dir
        self._jobs: dict = {}
        self._lock = threading.Lock()

    def create_job(self, job_id: str, input_path: str) -> dict:
        with self._lock:
            job = {
                "job_id": job_id,
                "status": "pending",
                "input_path": input_path,
                "log_lines": [],
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
        """Blocking — run in a daemon thread."""
        project_root = Path(__file__).parent.parent
        self._set_status(job_id, "running")
        try:
            if preprocess_fn is not None:
                self._append_log(job_id, "[INFO] Förbereder fil...")
                preprocess_fn(lambda msg: self._append_log(job_id, msg))
                self._append_log(job_id, "[INFO] Filkonvertering klar.")

            process = subprocess.Popen(
                [sys.executable, "cloud2entities.py", config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(project_root),
            )
            for line in process.stdout:
                self._append_log(job_id, line.rstrip())
            process.wait()
            status = "completed" if process.returncode == 0 else "failed"
        except Exception as exc:
            self._append_log(job_id, f"[ERROR] {exc}")
            status = "failed"

        self._set_status(job_id, status)

    # ── internal helpers ────────────────────────────────────────────────────

    def _set_status(self, job_id: str, status: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = status
                if status in ("completed", "failed"):
                    self._jobs[job_id]["finished_at"] = datetime.now().isoformat()

    def _append_log(self, job_id: str, line: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["log_lines"].append(line)
