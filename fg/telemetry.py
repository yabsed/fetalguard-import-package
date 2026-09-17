"""Private, append-only visit journal. Monitoring never changes training RNGs."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid

_active = None


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def emit(event, **fields):
    if _active is not None:
        _active.emit(event, **fields)


@contextmanager
def scope(name, **fields):
    start = time.monotonic()
    previous = _active.stage if _active else None
    if _active:
        _active.stage = name
    emit("stage_started", stage=name, **fields)
    try:
        yield
    except BaseException as exc:
        emit("stage_failed", stage=name, seconds=time.monotonic() - start, error_type=type(exc).__name__)
        raise
    else:
        emit("stage_finished", stage=name, seconds=time.monotonic() - start)
    finally:
        if _active:
            _active.stage = previous


class Tee:
    def __init__(self, stream, log, lock):
        self.stream, self.log, self.lock = stream, log, lock

    def write(self, message):
        with self.lock:
            result = self.stream.write(message)
            self.log.write(message)
            self.log.flush()
        return result

    def flush(self):
        with self.lock:
            self.stream.flush()
            self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def resource_sample(output):
    """Process CPU/RSS and device-wide GPU counters; unavailable is explicit."""
    usage = os.times()
    row = dict(timestamp=stamp(), process_cpu_seconds=usage.user + usage.system,
               process_rss_bytes=None, disk_free_bytes=shutil.disk_usage(output).free,
               gpu_status="unavailable", gpus=[], gpu_scope="device_wide_not_process")
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                row["process_rss_bytes"] = int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run([executable, "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                    timeout=2, check=True)
            for line in result.stdout.splitlines():
                index, utilization, used, total = line.split(",")
                row["gpus"].append(dict(index=int(index), utilization_percent=float(utilization),
                                        used_mib=float(used), total_mib=float(total)))
            row["gpu_status"] = "ok"
        except (OSError, ValueError, subprocess.SubprocessError):
            row["gpu_status"] = "query_failed"
    return row


class Visit:
    """Created before preflight/discovery, so those failures also leave evidence.

    Logs can contain IDs/paths. They are internal, never screened export files.
    Python stdout/stderr are mirrored; native C-library output may bypass Tee.
    """
    def __init__(self, output, cfg, interval=15, mode="analysis"):
        self.output, self.cfg = Path(output), cfg
        self.mode = mode
        self.id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
        self.path = self.output / "visits" / self.id / "internal" / "visit_audit"
        self.path.mkdir(parents=True)
        self.interval, self.stage, self.run, self.run_lock = interval, None, None, None
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.started = time.monotonic()

    def append(self, name, payload):
        with self.lock, (self.path / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

    def emit(self, event, **fields):
        payload = dict(timestamp=stamp(), elapsed_seconds=time.monotonic() - self.started,
                       event=event, stage=self.stage)
        payload.update(fields)
        self.append("events.jsonl", payload)

    def monitor(self):
        previous = None
        while not self.stop.is_set():
            try:
                row = dict(resource_sample(self.output), stage=self.stage)
                now = time.monotonic()
                row["process_cpu_percent_one_core"] = (
                    100 * (row["process_cpu_seconds"] - previous[1]) / (now - previous[0]) if previous else None)
                previous = (now, row["process_cpu_seconds"])
                self.append("resources.jsonl", row)
            except Exception as exc:
                self.emit("monitor_unavailable", error_type=type(exc).__name__)
            self.stop.wait(self.interval)

    def attach(self, run):
        self.run = Path(run)
        save_json(self.run / "visit_audit/attempts" / (self.id + ".json"),
                  dict(visit_id=self.id, path=os.path.relpath(self.path, self.run / "visit_audit")))
        self.emit("run_attached", run=str(self.run))
        self.pointer("running")

    def pointer(self, status):
        destination = self.run / "visit_audit" if self.run else self.path
        review = getattr(self, "review_index", None)
        save_json(self.output / "LATEST_VISIT.json", dict(visit_id=self.id, status=status, mode=self.mode,
                  audit=str(review or destination / "index.html"), internal_audit=str(destination / "index.html"),
                  export_audit=str(review) if review else None, journal=str(self.path),
                  scope="review_and_internal_separated" if review else "internal_only"))

    def export_diagnostics(self):
        from .export_diagnostics import build_export_diagnostics
        # Do not occupy a not-yet-built primary bundle after --check or a failed
        # early stage: run_export_review publishes that directory atomically.
        primary_ready = self.run is not None and (self.run.parent / "export_review/EXPORT_MANIFEST.json").is_file()
        root = self.run.parent if primary_ready and self.mode == "analysis" else self.path.parent.parent
        out = root / "export_review/visit_audit"
        # Every invocation gets an immutable snapshot; retain previous visit evidence.
        index = 2
        while out.exists():
            out = root / "export_review" / f"visit_audit_{index:03d}"
            index += 1
        self.review_index = build_export_diagnostics(self.run, out, self.cfg, attempt=self.path,
            images=self.mode == "analysis" and self.run is not None)
        latest = self.output / "LATEST.json"
        if self.mode == "analysis" and latest.is_file() and self.run is not None:
            value = json.loads(latest.read_text(encoding="utf-8"))
            if value.get("internal") == str(self.run):
                value["visit_audit"] = str(self.review_index)
                value["internal_visit_audit"] = str(self.run / "visit_audit/index.html")
                save_json(latest, value)
        print(f"반출 검토용 데이터·학습·방문 진단: {self.review_index}", flush=True)

    def refresh(self):
        try:
            from .visit_audit import build_visit_audit
            build_visit_audit(self.run, self.run / "visit_audit" if self.run else self.path, attempt=self.path)
        except Exception as exc:
            save_json(self.path / "audit_failure.json", dict(type=type(exc).__name__, error=str(exc)))
            print(f"방문 요약 생성 실패: {type(exc).__name__}. 원본 일지는 {self.path}", file=sys.stderr)

    def __enter__(self):
        global _active
        if _active is not None:
            raise RuntimeError("Visit journal already active")
        _active = self
        self.previous_sigterm = None
        if threading.current_thread() is threading.main_thread():
            self.previous_sigterm = signal.getsignal(signal.SIGTERM)
            def interrupted(signum, frame):
                raise KeyboardInterrupt("SIGTERM: visit interrupted")
            signal.signal(signal.SIGTERM, interrupted)
        self.stdout, self.stderr = sys.stdout, sys.stderr
        self.console = (self.path / "console.log").open("a", encoding="utf-8")
        sys.stdout, sys.stderr = Tee(self.stdout, self.console, self.lock), Tee(self.stderr, self.console, self.lock)
        save_json(self.path / "invocation.json", dict(visit_id=self.id, started=stamp(), config=self.cfg, mode=self.mode,
                  python=sys.version, pid=os.getpid(), privacy="internal_only_not_export_screened"))
        self.emit("visit_started")
        self.pointer("running")
        self.thread = threading.Thread(target=self.monitor, name="visit-resources", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, error, tb):
        global _active
        status = "interrupted" if kind and issubclass(kind, KeyboardInterrupt) else "failed" if kind else getattr(self, "analysis_status", "complete")
        try:
            self.stop.set()
            self.thread.join(timeout=3)
            if error is not None:
                save_json(self.path / "failure.json", dict(type=kind.__name__, error=str(error),
                          traceback="".join(traceback.format_exception(kind, error, tb))))
            self.emit("visit_finished", status=status)
            save_json(self.path / "status.json", dict(status=status, finished=stamp(),
                      seconds=time.monotonic() - self.started, run=str(self.run) if self.run else None))
            self.refresh()
            try:
                self.export_diagnostics()
            except Exception as exc:
                from .resilience import must_stop
                if must_stop(exc):
                    raise
                save_json(self.path / "export_diagnostics_failure.json", dict(type=type(exc).__name__, error=str(exc)))
                print(f"반출용 진단 생성 실패: {type(exc).__name__}. 내부 일지에서 원인을 확인하세요.", file=sys.stderr)
                if error is None:
                    if self.mode == "analysis" and self.run is not None and (self.run.parent / "status.json").is_file():
                        save_json(self.run / "export_diagnostics_failure.json", dict(type=type(exc).__name__, error=str(exc)))
                    status = "complete_with_issues"
                    save_json(self.path / "status.json", dict(status=status, finished=stamp()))
                    if self.mode == "analysis" and self.run is not None:
                        status_path = self.run.parent / "status.json"
                        value = json.loads(status_path.read_text()) if status_path.exists() else {}
                        value.update(status=status, diagnostics_status="failed")
                        save_json(status_path, value)
                        save_json(self.run / "pipeline_status.json", value)
            self.pointer(status)
        finally:
            if self.run_lock is not None:
                self.run_lock.close()
            sys.stdout, sys.stderr = self.stdout, self.stderr
            self.console.close()
            if self.previous_sigterm is not None:
                signal.signal(signal.SIGTERM, self.previous_sigterm)
            _active = None
        return False
