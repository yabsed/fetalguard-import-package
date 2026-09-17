"""Best-effort execution without inventing labels, metrics, or successful jobs.

Detailed exceptions are INTERNAL ONLY. Permission/storage failures and explicit
user interrupts are never swallowed. A failed unit must not publish its outputs
as if the whole unit succeeded; use local buffers or a rollback callback.
"""
from contextlib import contextmanager
import errno
import html
import json
from pathlib import Path
import traceback

from .telemetry import emit, save_json


class Unavailable(ValueError):
    """An analysis cannot be estimated with the available data/evidence."""


class ArtifactIntegrityError(ValueError):
    """Completed artifacts changed: do not overwrite them on resume."""


def must_stop(error):
    return isinstance(error, (PermissionError, ArtifactIntegrityError)) or (
        isinstance(error, OSError) and error.errno in {
            errno.ENOSPC, errno.EROFS, errno.EACCES, errno.EPERM, errno.EDQUOT})


class Issues:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        # Retain the append-only history, but summarize only this attempt.
        self.rows = []

    def record(self, job, error):
        if must_stop(error):
            raise error
        row = dict(job=str(job), type=type(error).__name__, error=str(error),
                   status="unavailable" if isinstance(error, Unavailable) else "failed",
                   traceback="".join(traceback.format_exception(type(error), error, error.__traceback__)))
        self.rows.append(row)
        with (self.out / "issues.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        save_json(self.out / "issues_summary.json", self.summary())
        emit("unit_failed", job=str(job), error_type=type(error).__name__)
        # IDs and raw messages stay in the internal journal, not public status.
        if len(self.rows) <= 3 or len(self.rows) % 100 == 0:
            print(f"일부 작업 미산출 {len(self.rows)}건 ({type(error).__name__}); 나머지 계속. 내부 기록: {self.out / 'issues.jsonl'}", flush=True)

    def summary(self):
        return dict(status="complete_with_issues" if self.rows else "complete",
                    failed_units=len(self.rows))

    @contextmanager
    def guard(self, job, rollback=None):
        try:
            yield
        except Exception as error:
            if must_stop(error):
                raise
            if rollback is not None:
                rollback()
            self.record(job, error)

    def finish(self, **fields):
        value = dict(self.summary(), **fields)
        save_json(self.out / "issues_summary.json", value)
        return value


def partial_report(run, out, cfg):
    """Always-readable internal landing page, independent of plotting/ML imports."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    coverage = Path(run) / "pipeline_status.json"
    value = json.loads(coverage.read_text(encoding="utf-8")) if coverage.exists() else {}
    rows = {name: row for name, row in value.get("stages", {}).items()
            if name not in {"report", "export_review", "onsite_figures"}}
    body = ["<!doctype html><meta charset='utf-8'><title>Partial CTG report</title>",
            "<h1>부분 완료 / Partial results</h1>",
            "<p>내부 전용. 미산출은 0이나 정상 판정이 아닙니다. 제외 내역과 성공한 분석만 검토하세요.</p>",
            "<p>라벨 자동 절단·보간 없음. 산모 분할 및 반출 심사 조건은 유지합니다.</p>",
            "<p><a href='../pipeline_status.json'>최종 단계별 상태 / pipeline status</a></p>", "<table>"]
    body += [f"<tr><td>{html.escape(name)}</td><td>{html.escape(row['status'])}</td></tr>"
             for name, row in rows.items()]
    body += ["</table><h2>Internal artifacts</h2><ul>"]
    for path in sorted(Path(run).rglob("*")):
        if path.is_file() and path.suffix in {".json", ".csv", ".jsonl", ".html"} and ".state" not in path.parts:
            relative = path.relative_to(run).as_posix()
            body.append(f"<li><a href='../{html.escape(relative, quote=True)}'>{html.escape(relative)}</a></li>")
    body.append("</ul>")
    (out / "report.html").write_text("\n".join(body), encoding="utf-8")
    (out / "report.md").write_text("# 부분 완료 / Partial results\n\n내부 전용. pipeline_status.json과 issues.jsonl을 확인하세요.\n", encoding="utf-8")
