"""Jupyter form: paths and buttons only; editing notebook code is unnecessary."""
from pathlib import Path
import subprocess
import sys
import threading
from datetime import datetime


def launch():
    import ipywidgets as widgets
    from IPython.display import display
    root = Path(__file__).resolve().parent
    data = widgets.Text(description="데이터 폴더", layout=widgets.Layout(width="95%"))
    output = widgets.Text(description="결과 폴더", value=str(root.parent / "03-실행결과"), layout=widgets.Layout(width="95%"))
    prior = widgets.Text(description="기존 산모 CSV", placeholder="선택: 이미 분석한 mother_id 목록 (최종 평가 제외)", layout=widgets.Layout(width="95%"))
    profile = widgets.Dropdown(description="모드", options=[("본선 전체 분석", "full"), ("빠른 mock test", "mock")], value="full")
    device = widgets.Dropdown(description="장치", options=["auto", "cpu", "cuda"])
    check = widgets.Button(description="사전검사", button_style="info")
    start = widgets.Button(description="전체 실행 / 이어서 실행", button_style="success", layout=widgets.Layout(width="230px"))
    stop = widgets.Button(description="실행 중단", button_style="warning", disabled=True)
    status = widgets.HTML("PyTorch Python 3.10+ 커널에서 데이터 폴더를 입력하고 실행하세요.")
    log = widgets.Textarea(value="", disabled=True, layout=widgets.Layout(width="100%", height="420px"))
    active = {"process": None}

    def run(check_only=False):
        if not data.value.strip() or not output.value.strip():
            status.value = "데이터 폴더와 결과 폴더를 모두 입력하세요."
            return
        start.disabled = check.disabled = True
        stop.disabled = False
        command = [sys.executable, "-u", "-B", str(root / "run.py"), "--data", data.value.strip(),
                   "--output", output.value.strip(), "--profile", profile.value, "--device", device.value]
        if check_only:
            command.append("--check")
        if prior.value.strip():
            command += ["--prior-cohort", prior.value.strip()]
        status.value = "실행 중 — 로그를 확인하세요."

        def worker():
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=root)
                active["process"] = process
                for line in process.stdout:
                    log.value = (log.value + line)[-50000:]
                code = process.wait()
                status.value = ("사전검사 완료 — 학습은 실행하지 않았습니다." if check_only else
                    "완료 — export_review/onsite_figures/index.html에서 결과를 읽으세요. 선별 집계 CSV는 export_review/csv/, 이미지 심사 후보는 export_review/images/입니다.") if code == 0 else f"중단/실패 (exit {code}). 마지막 오류를 확인하세요. 완료된 학습은 재실행 시 재사용됩니다."
            except Exception as exc:
                log.value += str(exc)
                status.value = "실행 실패"
            finally:
                active["process"] = None
                start.disabled = check.disabled = False
                stop.disabled = True
        threading.Thread(target=worker, daemon=True).start()

    def terminate(_):
        if active["process"] is not None:
            active["process"].terminate()
    start.on_click(lambda _: run(False))
    check.on_click(lambda _: run(True))
    stop.on_click(terminate)
    display(widgets.VBox([widgets.HTML("<h3>CTG 본선 원클릭 실행 · 설계 2.0</h3><p>코드 수정 없이 경로와 모드만 입력합니다. 현장 결과와 반출 심사용 집계 묶음을 별도로 생성합니다.</p>"),
        data, output, widgets.HBox([profile, device]), widgets.HBox([check, start, stop]), prior, status, log]))
