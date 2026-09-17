# `02-반입할-파일` 현장 실행 매뉴얼

> 연속 실행 업데이트: `data` 이후 개별 데이터·모델 오류는 기록·제외하고 가능한 작업을 계속합니다.
> `complete_with_issues`는 제외 또는 미산출이 있는 부분 완료이며 전체 성공이 아닙니다.
> `internal/data/record_exclusions.csv`, `internal/pipeline_status.json`, 각 단계 `issues.jsonl`을 확인하세요.
> 검증기는 부분 완료에 `PARTIAL_VALIDATED`를 표시합니다. 아래의 엄격 중단 예시는 현재 레코드별 제외로 처리될 수 있습니다.
> 환경검사, 권한·저장공간·무결성 검사와 기관 반출 승인 조건은 유지됩니다. 환경 설치 방식은 이번 수정에서 변경하지 않았습니다.

이 문서는 Python과 머신러닝에 익숙하지 않은 실행 담당자가, 현장에서 처음 보는 PC와 처음 보는 데이터를 만나도 순서를 잃지 않도록 만든 운영 절차서다.

이 패키지의 안전한 실행 순서는 다음과 같다.

```text
Python 위치 확인
    ↓
데이터 조사만 실행 (--survey-only)
    ↓ 조사 결과가 입력 계약과 대체로 맞는가?
환경·공식 모델 사전검사 (--check)
    ↓ 사전검사가 통과했는가?
짧은 연결 시험 (--profile mock)
    ↓ mock이 끝까지 완료됐는가?
본 분석 (--profile full)
    ↓ status.json이 complete인가?
결과 검증·내부 보존·기관 반출 심사
```

모르는 상태에서 곧바로 `full`을 실행하지 않는다. 현장에서 최소한 **조사 → 사전검사 → mock → full** 순서를 지키는 것이 이 문서의 핵심이다.

---

## 1. 가장 짧은 실행 요약

Linux 기준으로, 아래 네 명령을 순서대로 실행한다. `/경로/...`는 실제 경로로 바꾸고 경로 전체를 큰따옴표로 감싼다.

```bash
cd "/안심존/02-반입할-파일"

python -B run.py --survey-only \
  --data "/안심존/압축해제한-데이터-상위폴더" \
  --output "/팀폴더/분석결과"

python -B run.py --check \
  --data "/안심존/압축해제한-데이터-상위폴더" \
  --output "/팀폴더/분석결과" \
  --profile mock --device auto

python -B run.py \
  --data "/안심존/압축해제한-데이터-상위폴더" \
  --output "/팀폴더/분석결과" \
  --profile mock --device auto

python -B run.py \
  --data "/안심존/압축해제한-데이터-상위폴더" \
  --output "/팀폴더/분석결과" \
  --profile full --device auto
```

`python`이 올바른 실행 환경이 아닐 수 있다. 위 명령을 실행하기 전에 반드시 다음 절에서 Python을 먼저 확인한다.

완료 판정은 콘솔의 마지막 문장이나 폴더 존재 여부가 아니라 다음 세 가지로 한다.

1. `분석결과/full-실행해시/status.json`의 `status`가 `complete`이다.
2. `tools/validate_run.py` 검증이 `PASS`이다.
3. `export_review/onsite_figures/index.html`과 내부 보고서가 실제로 열린다.

`mock`은 프로그램 연결 시험일 뿐이다. `mock`의 성능 숫자를 연구 결과로 사용하면 안 된다.

---

## 2. 먼저 알아야 할 네 개의 폴더

현장에서는 아래 네 위치를 종이나 메모장에 먼저 적는다.

| 이름 | 뜻 | 규칙 |
|---|---|---|
| 패키지 폴더 | 이 문서와 `run.py`가 있는 `02-반입할-파일` | 가능하면 압축 해제 후 수정하지 않는다. |
| 데이터 폴더 | `annotation_person`, `labels`, `emr` 등을 아래에서 찾을 수 있는 상위 폴더 | 읽기 전용으로 취급한다. 압축 파일 자체가 아니라 압축을 푼 폴더를 지정한다. |
| 결과 폴더 | 실행 결과와 일지가 쌓이는 폴더 | 패키지 폴더 밖이며 데이터 폴더 밖이어야 한다. 쓰기 권한과 여유 공간이 필요하다. |
| Python 실행 파일 | 필요한 패키지가 설치된 Python | Jupyter 커널과 터미널 Python이 서로 다를 수 있다. 정확한 위치를 기록한다. |

잘못된 예:

```text
02-반입할-파일/data              # 패키지 안에 원천 데이터를 넣음
원천데이터/results               # 데이터 안에 결과를 저장함
02-반입할-파일/results           # 패키지 안에 결과를 저장함
data.zip                         # 압축 파일을 --data로 지정함
```

권장 예:

```text
/안심존/import/02-반입할-파일
/안심존/source/korean-ctg
/안심존/team/fetalguard-results
```

경로에 한글이나 공백이 있어도 되지만, 터미널 명령에서는 항상 `"전체 경로"`처럼 큰따옴표를 사용한다.

---

## 3. 현장에 가기 전에 준비할 것

이 패키지는 현장에서 패키지를 설치하거나 모델을 다운로드하지 않는다. 실행 중 과학 계산 단계의 네트워크 연결도 차단한다. 가상환경, Conda 환경, Docker 이미지는 패키지에 포함되어 있지 않다.

따라서 출발 전에 다음을 준비해야 한다.

- 기관에 Python 3.10 이상과 [requirements-runtime.txt](requirements-runtime.txt)의 패키지 사용 가능 여부를 확인한다.
- 가능하면 현장과 같은 OS·Python·GPU 구성에서 `mock`을 한 번 끝까지 실행한다.
- 승인된 최종 ZIP을 만들 때 `tools/seal_package.py`로 무결성 목록을 갱신한다. 개발 중 파일을 바꾼 뒤 봉인하지 않으면 현장에서 `반입 파일 무결성 불일치`로 중단된다.
- 데이터 위치, 결과 저장 위치, 결과 폴더의 퇴실 후 보존 정책을 기관 담당자에게 확인한다.
- GPU 사용 가능 여부뿐 아니라 예약 시간, 전원 정책, 저장 공간, 장시간 작업 유지 가능 여부를 확인한다.
- 현장에서 패키지를 새로 설치할 수 있다고 가정하지 않는다. 인터넷, 관리자 권한, 컴파일러가 없을 수 있다.
- 이전 방문에서 이미 분석한 산모 목록을 사용할 계획이면, 같은 ID 체계의 `mother_id` 열을 가진 CSV를 별도로 승인받아 준비한다.

패키지 개발 담당자가 최종 ZIP을 만드는 명령은 다음과 같다. 이 명령은 **현장 실행 명령이 아니다**.

```bash
python -B tools/seal_package.py --zip ../02-반입할-파일-final.zip
```

최종 ZIP을 만든 뒤에는 그 ZIP을 다시 풀어 `--survey-only`, `--check`, `mock`을 시험하는 것이 가장 확실하다. 원본 데이터가 없는 사전 준비 장소에서는 승인된 테스트 데이터로 시험한다.

---

## 4. 어떤 Python을 사용해야 하나

### 4.1 선택 기준

우선순위는 다음과 같다.

1. 기관이 제공한 **PyTorch / Python 3.10 이상 Jupyter 커널**
2. 기관이 제공한, [requirements-runtime.txt](requirements-runtime.txt)가 설치된 Conda 또는 venv Python
3. 담당자가 사전에 검증한 Python 실행 파일
4. 단순 Python 3.10 이상: `--survey-only`에는 사용할 수 있지만, 전체 분석에는 필요한 패키지가 없을 수 있음

Python 3.10 이상이라는 사실만으로 전체 분석 환경이 맞다는 뜻은 아니다. 실제 전체 분석에는 NumPy, pandas, SciPy, scikit-learn, CatBoost, XGBoost, PyTorch, Matplotlib, Pillow, OpenCV, tabulate가 필요하다. 버전 기준은 [requirements-runtime.txt](requirements-runtime.txt)에 있다.

코드는 Python 3.10 이상을 검사하고 필요한 모듈을 불러와 보지만, 모든 미래 버전 조합의 호환성을 보장하지는 않는다. 가능하면 사전 승인·검증된 버전 조합을 사용한다.

### 4.2 터미널에서 확인

Linux/macOS:

```bash
python --version
python -c "import sys, platform; print(sys.executable); print(sys.version); print(platform.platform())"
```

`python`이 없으면 `python3`도 같은 방식으로 확인한다.

Windows 명령 프롬프트:

```bat
where python
python --version
python -c "import sys, platform; print(sys.executable); print(sys.version); print(platform.platform())"
```

Windows PowerShell:

```powershell
Get-Command python
python --version
python -c "import sys, platform; print(sys.executable); print(sys.version); print(platform.platform())"
```

출력된 `sys.executable`을 실행 기록에 복사한다. 여러 Python이 보이면 임의로 하나를 고르지 말고, PyTorch가 준비된 환경의 실행 파일을 사용한다.

특정 Python을 직접 지정하는 예:

```bash
"/opt/conda/envs/approved/bin/python" -B run.py --help
```

```bat
"C:\approved-env\python.exe" -B run.py --help
```

### 4.3 Jupyter에서 확인

노트북 셀에서 다음을 실행한다.

```python
import sys
print(sys.executable)
print(sys.version)
```

`START.ipynb`의 버튼은 바로 이 `sys.executable`로 `run.py`를 실행한다. 터미널의 `python`과 결과가 달라도 이상한 일이 아니다.

### 4.4 환경별 판단

| 발견한 환경 | 할 일 |
|---|---|
| 승인된 PyTorch Python 3.10+ 환경 | `--survey-only` 후 `--check`로 진행한다. |
| Python 3.10+만 있고 ML 패키지가 없음 | 우선 `--survey-only`만 실행한다. 현장에서 임의 설치하지 말고 승인된 환경을 요청한다. |
| Python이 여러 개임 | 각각의 `sys.executable`을 확인하고 승인된 환경을 명시적으로 사용한다. |
| Python 3.10 미만만 있음 | 전체 분석을 시도하지 않는다. 환경 정보와 조사 가능 여부를 담당자에게 전달한다. |
| CUDA가 없거나 동작하지 않음 | `--device auto` 또는 `--device cpu`로 mock 가능 여부를 확인한다. full의 소요 시간이 방문 시간을 넘을 수 있으므로 일정을 다시 판단한다. |

현장에서 `pip install`로 즉석 수리하는 것은 마지막 수단도 아니다. 기관 승인, 인터넷, 바이너리 호환성, 재현성 문제가 모두 생긴다. 환경이 없으면 원천 조사와 증거 보존까지 수행한 뒤 승인된 환경을 준비하는 편이 낫다.

---

## 5. 데이터가 무엇인지 모를 때

### 5.1 데이터 폴더의 최소 구조

프로그램은 지정한 상위 폴더 아래를 재귀적으로 찾는다. `Training/Validation/Test`로 나뉘어 있어도 괜찮다. 중요한 것은 실제 파일의 **바로 위 폴더 이름**이다.

```text
선택할-데이터-상위폴더/
├── .../annotation_person/*.json   # 필수: FHR/TOCO 수치 신호
├── .../labels/*.json              # 필수: 구간 라벨
├── .../emr/*.json                 # 주 분석에 사실상 필요: 안전한 산모 단위 분할
└── .../refine_images/*.png        # 선택: Bbox/공식 YOLO 평가
```

`annotation_person`과 `labels`가 없으면 분석은 시작할 수 없다. EMR 파일이 없거나 `Mother.de-identification_ID`가 비어 있으면 레코드 ID로 대신하지 않고 주 분석을 중단한다. 이는 같은 산모가 train과 test에 섞이는 누수를 막기 위한 안전장치다.

주요 수치 신호 계약은 다음과 같다.

- 한 구간은 5분이다.
- FHR와 TOCO는 각 150개 표본이다.
- 표본 간격은 2초이다.
- 라벨 길이와 선택된 태아의 구간 수가 맞아야 한다.
- 쌍태아 ID와 `twin1`/`twin2` 구조가 맞아야 한다.
- 동일 ID에 내용이 다른 중복 파일이 있으면 중단한다.

형식이 다를 때 프로그램이 조용히 길이를 맞추거나 산모 ID를 만들어내지 않는다. 이 실패는 데이터 어댑터나 기관 확인이 필요하다는 증거다.

### 5.2 첫 명령은 항상 조사 전용

조사 전용 실행은 학습과 GPU 검사를 하지 않는다. 표준 라이브러리 중심으로 파일 구성, JSON 필드, 연결 관계, 신호 길이·간격·0·평탄 구간·비정상 값을 조사한다.

```bash
python -B run.py --survey-only \
  --data "/안심존/압축해제한-데이터-상위폴더" \
  --output "/팀폴더/분석결과"
```

이 작업도 데이터 전량을 순회하므로 파일이 많거나 저장장치가 느리면 시간이 걸린다. 원천 파일을 수정하지는 않지만, 결과 일지에는 원본의 상대 경로와 ID가 들어갈 수 있으므로 결과 폴더는 내부 자료로 다룬다.

완료 후 `/팀폴더/분석결과/LATEST_VISIT.json`을 연다. `export_audit`는 생성된 반출 심사용 집계 보고서, `internal_audit`는 내부 상세 보고서, `journal`은 원본 로그·조사 파일이 있는 폴더다. `audit`은 반출용 보고서가 생성됐으면 그것을, 아니면 내부 보고서를 가리킨다. 파일 탐색기에서 해당 `index.html`을 웹 브라우저로 열어도 인터넷은 필요하지 않다.

### 5.3 조사 보고서에서 볼 것

아래 질문에 답을 적는다.

- `annotation_person`, `labels`, `emr` 파일이 실제로 발견됐는가?
- annotation ID 중 label과 EMR이 모두 연결되는 수는 얼마인가?
- JSON 파싱 오류나 읽기 오류가 있는가?
- 2초 간격, FHR 150개, TOCO 150개인 후보 구간은 얼마나 되는가?
- 다른 길이·간격·자료형 변종이 있는가?
- 동일 ID 중복과 서로 다른 ID의 동일 신호 표현이 있는가?
- FHR/TOCO의 0, 음수, 비수치, 긴 평탄 구간이 특정 기관 코드에 몰려 있는가?
- `0`, 빈칸, `9999`의 의미를 기관에 확인했는가?
- 측정 시각, 장비, 단위, 판독자 합의 과정은 확인됐는가?

조사 보고서의 수는 **원천 파일/원천 구간 기준**이다. 최종 선택 태아, 결측 제외, 산모 단위 분할 이후의 학습 표본 수와 다르다. 조사에서 파일이 연결됐다고 해서 분석 적격 판정이 끝난 것도 아니다.

다음 중 하나면 바로 full로 넘어가지 않는다.

- 필수 폴더가 발견되지 않음
- JSON 파싱 오류가 다수 발생함
- 예상하지 않은 간격·길이 변종이 있음
- annotation과 label/EMR 연결이 크게 끊김
- 산모 ID의 뜻이나 보존 여부가 불명확함
- `0`, `9999`, 단위, 라벨 의미가 불명확함
- 서로 충돌하는 중복 파일이 의심됨

이 경우 `LATEST_VISIT.json`, 조사 HTML, `survey/summary.json`, `survey/issues.jsonl`, `survey/shape_variants.csv`를 내부에 보존하고 데이터 담당자와 확인한다. 원천 JSON을 현장에서 임의 편집하지 않는다.

---

## 6. 환경·모델 사전검사

조사 결과가 진행 가능하면 다음 명령을 실행한다.

```bash
python -B run.py --check \
  --data "/안심존/압축해제한-데이터-상위폴더" \
  --output "/팀폴더/분석결과" \
  --profile mock --device auto
```

사전검사는 다음을 확인한다.

- 반입 패키지 파일의 SHA256 무결성
- 원천 데이터 조사
- Python 3.10 이상
- 필수 Python 모듈 import
- PyTorch와 NumPy의 기본 연동
- CUDA 요청 시 CUDA 사용 가능 여부
- 동봉된 공식 XGBoost 모델 로딩
- 동봉된 공식 YOLO TorchScript 모델의 실제 추론 1회
- 입력 파일 발견과 해시 계산

따라서 `--check`도 데이터 전체 조사와 입력 파일 해시 때문에 시간이 걸릴 수 있다. `CHECK OK`가 나와야 학습 단계로 넘어간다. `--check`와 `--survey-only`는 함께 쓰지 않는다.

`--device auto`는 CUDA가 실제로 사용 가능하면 GPU, 아니면 CPU를 선택한다. 현장에서는 우선 `auto`가 가장 안전하다. `--device cuda`는 반드시 GPU를 써야 할 때만 사용하며, CUDA를 못 찾으면 중단한다.

`--check`는 학습을 하지 않는다. 분석 완료를 뜻하는 `status.json`도 만들지 않는다. 최신 점검 일지 위치는 `LATEST_VISIT.json`에서 확인한다.

---

## 7. mock으로 전체 연결 시험

사전검사가 통과하면 같은 Python, 데이터, 결과 폴더로 mock을 실행한다.

```bash
python -B run.py \
  --data "/안심존/압축해제한-데이터-상위폴더" \
  --output "/팀폴더/분석결과" \
  --profile mock --device auto
```

mock도 원천 데이터는 전량 읽는다. 대신 다음 계산 예산을 줄인다.

- CV 2-fold × seed 1개
- 트리 최대 30회
- CNN width 2개 × seed 1개 × 정규화 2개, 최대 3 epoch
- bootstrap 50회

따라서 mock은 다음을 확인하는 용도다.

- 엄격한 데이터 계약을 실제로 통과하는지
- 산모 단위 분할이 만들어지는지
- 트리 모델과 CNN이 실제로 학습되는지
- 공식 모델 평가와 보고서 생성까지 끝나는지
- 결과 폴더 권한과 디스크 사용이 문제없는지

mock이 성공하면 결과 폴더 아래에 `mock-16자리해시/`가 생긴다. 다음처럼 검증한다.

```bash
python -B tools/validate_run.py "/팀폴더/분석결과/mock-16자리해시"
```

마지막 출력의 `status`가 `PASS`인지 확인한다. mock 결과의 AP, AUROC, CI는 실행 시험 숫자이며 연구 결론으로 보고하지 않는다.

---

## 8. full 본 분석

mock이 끝까지 통과했고 시간·전원·저장 공간이 확보됐을 때만 full을 실행한다.

```bash
python -B run.py \
  --data "/안심존/압축해제한-데이터-상위폴더" \
  --output "/팀폴더/분석결과" \
  --profile full --device auto
```

기본 thread 수는 4다. 기관이 허용한 CPU 수가 더 적으면 다음처럼 명시한다.

```bash
python -B run.py \
  --data "/안심존/압축해제한-데이터-상위폴더" \
  --output "/팀폴더/분석결과" \
  --profile full --device auto --threads 2
```

thread 수, 장치, Python 환경, 패키지 버전, 입력 파일 또는 설정이 바뀌면 실행 식별 해시도 달라질 수 있다. 재개하려면 처음과 **같은 명령, 같은 Python, 같은 데이터, 같은 패키지**를 사용한다.

full은 GPU에서도 오래 걸릴 수 있고, CPU에서는 현장 시간을 크게 넘길 수 있다. mock 시간만 비례 확대해 full 시간을 단정하지 않는다. 실행 중 `visits/.../resources.jsonl`에 CPU/RAM과 가능한 GPU/VRAM 표본이 15초마다 기록된다.

### 이전에 분석한 산모가 있는 경우

정확히 같은 산모 ID 체계의 CSV가 있고, CSV에 `mother_id` 열이 있을 때만 사용한다.

```bash
python -B run.py \
  --data "/안심존/압축해제한-데이터-상위폴더" \
  --output "/팀폴더/분석결과" \
  --profile full --device auto \
  --prior-cohort "/안심존/내부/기존산모.csv"
```

이 목록의 산모는 주 holdout의 validation/test에서 제외되고 train에만 들어간다. CV, H5, LOSO까지 모두 새 미관측 평가로 만들어 주는 기능은 아니다. 목록의 출처나 ID 체계가 불확실하면 추측해서 사용하지 말고, 미제공 상태를 기록한다.

---

## 9. Jupyter로 실행하는 방법

기관의 JupyterLab과 `ipywidgets`가 정상 동작하면 초보자에게는 `START.ipynb`가 편하다.

1. JupyterLab에서 승인된 PyTorch / Python 3.10 이상 커널을 선택한다.
2. `START.ipynb`를 연다.
3. **Run All**을 누른다.
4. 데이터 폴더와 결과 폴더를 입력한다.
5. 장치는 `auto`로 둔다.
6. 먼저 **데이터 조사만**을 누른다.
7. 조사 결과를 확인한 뒤 **사전검사**를 누른다.
8. 모드를 `빠른 mock test`로 바꾸고 **전체 실행 / 이어서 실행**을 누른다.
9. mock 검증 후 모드를 `본선 전체 분석`으로 바꾸고 실행한다.

브라우저 창을 닫거나 노트북 커널을 재시작하면 실행을 잃을 수 있다. 장시간 full은 기관의 세션 유지 정책을 먼저 확인한다. 버튼 화면이 나타나지 않거나 위젯 오류가 나면 코드 셀을 고치지 말고 터미널 방식으로 전환한다.

---

## 10. Windows에서 실행하는 방법

### 명령 프롬프트

```bat
cd /d "D:\반입\02-반입할-파일"

python -B run.py --survey-only ^
  --data "D:\원천데이터\korean-ctg" ^
  --output "D:\팀폴더\fetalguard-results"

python -B run.py --check ^
  --data "D:\원천데이터\korean-ctg" ^
  --output "D:\팀폴더\fetalguard-results" ^
  --profile mock --device auto

python -B run.py ^
  --data "D:\원천데이터\korean-ctg" ^
  --output "D:\팀폴더\fetalguard-results" ^
  --profile mock --device auto
```

full은 마지막 명령의 `mock`을 `full`로 바꾼다. `RUN.cmd`도 같은 `run.py`를 실행하지만, 어떤 Python이 선택됐는지 분명히 하기 위해 처음에는 위 명령을 권장한다.

특정 Python 실행 파일을 사용할 때:

```bat
"C:\approved-env\python.exe" -B run.py --help
```

### PowerShell

PowerShell에서 실행 파일 전체 경로를 쓸 때는 앞에 `&`를 붙인다.

```powershell
Set-Location "D:\반입\02-반입할-파일"
& "C:\approved-env\python.exe" -B run.py --survey-only `
  --data "D:\원천데이터\korean-ctg" `
  --output "D:\팀폴더\fetalguard-results"
```

---

## 11. 중단됐을 때 재개하는 법

같은 실행을 재개하려면 처음과 같은 명령을 다시 실행한다.

```bash
python -B run.py \
  --data "/안심존/압축해제한-데이터-상위폴더" \
  --output "/팀폴더/분석결과" \
  --profile full --device auto
```

완료된 단계는 저장된 해시를 검증한 뒤 재사용한다. 일부 CNN 학습은 마지막 체크포인트에서 이어질 수 있다.

재개 전에 지켜야 할 규칙:

- 같은 실행을 두 터미널에서 동시에 시작하지 않는다.
- 결과 파일을 수동으로 이름 변경·수정·삭제하지 않는다.
- 원천 데이터 파일을 바꾸지 않는다.
- Python 환경이나 명령 옵션을 바꾸지 않는다.
- 기존 결과를 다른 폴더에 부분 복사한 뒤 재개하지 않는다.

완료 단계의 파일이 변조되거나 삭제되면 프로그램은 재사용을 거부한다. 이때는 원본 결과를 복구하거나, 담당자와 협의해 새로운 결과 루트에서 다시 실행한다.

`Ctrl+C`나 UI의 중단 버튼처럼 정상적인 중단 신호를 쓰면 실패/중단 요약과 진단을 남기려고 시도한다. 전원 차단이나 강제 종료(`kill -9`)는 최종 요약을 남기지 못하며 상태가 `running`으로 남을 수 있다. 그래도 이미 저장된 일지와 체크포인트는 삭제하지 않는다.

---

## 12. 성공과 실패를 판정하는 법

### 12.1 full 성공

결과 구조는 대략 다음과 같다.

```text
분석결과/
├── LATEST.json
├── LATEST_VISIT.json
├── visits/                         # 모든 시도의 조사·콘솔·자원 일지
└── full-16자리해시/
    ├── status.json                 # status == complete 확인
    ├── internal/                   # 원천 ID/예측/모델을 포함할 수 있는 내부 전용
    └── export_review/              # 현장 검토와 반출 심사 준비용
```

검증 명령:

```bash
python -B tools/validate_run.py "/팀폴더/분석결과/full-16자리해시"
```

확인 순서:

1. `full-.../status.json`: `status`가 `complete`
2. 검증 명령: `PASS`
3. `full-.../export_review/onsite_figures/index.html`: 현장 결과를 질문 순서로 검토
4. `full-.../internal/report/report.html`: 내부 상세 보고서
5. `LATEST.json`의 `visit_audit` 경로: 데이터·학습·시간·자원·다음 방문 진단. 첫 생성은 `full-.../export_review/visit_audit/index.html`이며, 재실행 시 `visit_audit_002` 등으로 추가되므로 포인터가 가리키는 최신 파일을 연다.
6. `LATEST.json`: 위 최신 분석 위치를 가리키는 포인터
7. `LATEST_VISIT.json`: 가장 최근 실행 시도의 진단 위치

`LATEST.json`이 존재한다는 이유만으로 성공한 것은 아니다. 분석 시작 때 먼저 만들어지므로 반드시 `status.json`과 검증 결과를 확인한다.

### 12.2 실패

실패 시 먼저 `/팀폴더/분석결과/LATEST_VISIT.json`을 연다. 그 안의 **`journal` 경로**(보통 `visits/<방문ID>/internal/visit_audit/`)에서 다음을 확인한다. `audit`이나 `export_audit`이 가리키는 반출용 요약에는 원본 로그가 없다.

- `index.html`: 사람이 읽는 요약
- `console.log`: Python 표준 출력과 오류
- `failure.json`: 실패 종류와 traceback
- `events.jsonl`: 어느 단계에서 실패했는지
- `resources.jsonl`: 실행 당시 자원 표본
- `survey/`: 데이터 조사 결과

분석 상태 파일이 생성된 뒤 실패했다면 `full-.../status.json`의 `details` 경로도 확인한다. 일반 분석 실패는 `internal/failure.json`, 반출용 진단 생성 실패는 `internal/export_diagnostics_failure.json`에 기록된다.

오류 메시지의 마지막 한 줄만 복사하지 말고, 위 폴더 전체를 내부에 보존한다. 경로와 ID가 포함될 수 있으므로 승인 없이 외부 메신저나 개인 저장장치로 보내지 않는다.

---

## 13. 자주 만나는 오류와 조치

| 증상 또는 메시지 | 뜻 | 해야 할 일 |
|---|---|---|
| `반입 파일 무결성 불일치` | 봉인 뒤 패키지 파일이 바뀌었거나 복사가 손상됨 | 현장에서 코드를 고치지 말고 승인된 원본 ZIP을 다시 푼다. 개발본이라면 출발 전에 다시 봉인한다. |
| `데이터 폴더 없음` | `--data` 경로가 틀림 | 압축을 푼 상위 폴더의 실제 경로를 확인한다. |
| `annotation_person/*.json 및 labels/*.json을 찾지 못했습니다` | 너무 아래/위의 엉뚱한 폴더를 골랐거나 폴더명이 다름 | 조사 결과와 실제 폴더 트리를 확인하고 올바른 상위 폴더를 지정한다. 구조가 다르면 어댑터 개발 대상으로 기록한다. |
| `환경 사전검사 실패. 자동 설치하지 않습니다` | 잘못된 Python이거나 필수 패키지가 없음 | `sys.executable`을 다시 확인하고 승인된 PyTorch 환경으로 바꾼다. 즉석 설치하지 않는다. |
| `CUDA requested but unavailable` | `--device cuda`인데 PyTorch가 CUDA를 못 씀 | 우선 `--device auto`로 확인한다. full CPU 실행 시간이 허용되는지는 별도로 판단한다. |
| 공식 XGBoost/YOLO 로딩 실패 | 모델 파일 손상 또는 라이브러리 비호환 가능성 | 패키지 무결성, Python/패키지 버전을 확인한다. 임의로 공식 모델을 끄지 않는다. |
| `Missing mother ID` | 안전한 산모 단위 분할을 만들 수 없음 | 레코드 ID로 대신하지 않는다. 기관에 `Mother.de-identification_ID` 의미와 누락을 확인한다. |
| `Label ID mismatch`, 라벨 길이 불일치 | 파일명/ID/구간 정렬 계약이 맞지 않음 | 원천 파일을 수동 수정하지 말고 해당 ID와 조사 결과를 데이터 담당자에게 전달한다. |
| `expected 300s`, `0.5Hz / 150 samples` | 5분·2초·150표본 계약과 다름 | `shape_variants.csv`로 규모를 파악하고, 다음 반입 버전에 명시적 어댑터나 별도 분석 설계를 준비한다. |
| `Bbox 검증 ... 불일치` | 라벨 박스와 이미지의 위치/크기가 맞지 않음 | `bbox_audit.csv`를 확인한다. 단순히 `strict_bbox=false`로 바꾸지 말고 분석 책임자의 결정을 받는다. |
| `Conflicting duplicate ...` | 같은 ID에 내용이 다른 파일이 둘 이상 있음 | 어느 배포본이 정본인지 기관에 확인한다. 임의 선택·삭제하지 않는다. |
| `같은 실행이 이미 진행 중입니다` | 동일 실행을 다른 프로세스가 사용 중 | 다른 터미널/Jupyter 작업을 확인한다. 실행 중이면 기다리고, 강제 종료부터 하지 않는다. |
| `완료 단계 산출물이 변경/삭제됨` | 재개 대상 결과가 손상 또는 수정됨 | 원본을 복구하거나 새 결과 폴더에서 재실행한다. |
| `No space left on device` 또는 쓰기 오류 | 결과 저장 공간/권한 부족 | 실행을 무리하게 계속하지 말고 내부 결과 전체를 보존한 뒤 승인된 새 결과 위치를 마련한다. |
| 실행이 느려 보임 | 전량 조사·해시·부트스트랩·CNN이 진행 중일 수 있음 | `console.log`, `events.jsonl`, `resources.jsonl`의 최근 갱신을 확인한다. 파일이 갱신 중이면 임의 종료하지 않는다. |

설정을 바꿔 실패를 숨기지 않는다. 특히 `strict_bbox`, `official_models`, `cnn`, 데이터 제외 기준, seed, 학습 예산을 바꾸는 것은 단순한 기술 수리가 아니라 분석 설계 변경이다.

---

## 14. 현장 시간이 부족할 때의 우선순위

방문 시간 안에 full을 끝내지 못할 수도 있다. 그때 우선순위는 다음과 같다.

### 반드시 확보할 것

1. `--survey-only` 결과
2. 실제 사용 가능한 Python 경로와 버전
3. `--check` 성공/실패 결과
4. `LATEST_VISIT.json`이 가리키는 일지 전체
5. 데이터 담당자에게 확인할 질문과 답변

### 가능하면 확보할 것

1. mock 완주와 `validate_run.py` 결과
2. mock의 단계별 시간·CPU/RAM/GPU 기록
3. full 실행 시작 또는 중단 가능한 체크포인트

### 끝까지 완료할 것

1. full의 `status.json == complete`
2. `validate_run.py`의 `PASS`
3. 현장 보고서 검토
4. 결과 루트 전체의 내부 보존

첫 방문에서 full이 실패해도 조사를 제대로 남겼다면 완전히 실패한 방문은 아니다. 입력 변종, 환경 차이, 예상 시간, 기관에 물어볼 내용을 근거로 다음 반입 버전을 준비할 수 있다.

---

## 15. 결과 보존과 반출

퇴실 전에 결과 루트 전체를 내부 저장소에 보존한다.

```text
/팀폴더/분석결과/
├── visits/              # 반드시 보존: 실패 전·후를 포함한 실행 일지
├── mock-.../            # 실행 시험
├── full-.../            # 본 분석
├── LATEST.json
└── LATEST_VISIT.json
```

`full-.../internal/`에는 원본 경로, 식별자, 개인별 예측, 신호, 모델 가중치가 포함될 수 있다. `visits/<방문ID>/internal/`의 로그와 원천 조사 CSV에도 ID와 경로가 포함될 수 있다. 이 `internal/` 자료는 내부 전용이다.

조사 전용·사전검사·주 심사 묶음 생성 전 실패의 선별 집계는 `visits/<방문ID>/export_review/visit_audit/`에 생성된다. 이 폴더는 반출 심사 후보이며, 정확한 최신 위치는 `LATEST_VISIT.json`의 `export_audit`에서 확인한다. 조사 전용·사전검사는 HTML·CSV를 생성하며 PNG는 생성하지 않는다. `visits/` 전체를 복사하지 말고 그 아래 `export_review/`의 승인된 자료만 선택한다.

`export_review/`라는 이름은 자동 반출 승인을 뜻하지 않는다. 기본 상태는 `pending_institution_review`다.

- 현장 이해 시작점: `export_review/onsite_figures/index.html`
- 성능 이미지 심사 후보: `export_review/images/`의 PNG
- 추가 진단 이미지 심사 후보: 최신 진단 폴더의 `images/` PNG (`visit_audit/`, 재실행 시 `visit_audit_002/` 등)
- CSV 심사 후보: 각 검토 폴더의 `csv/`
- 현장 전용이며 자동 반출 후보가 아닌 것: `export_review/onsite_figures/`

이미지 전용 제출에서는 승인된 `images/`의 PNG만 고른다. 상위 `export_review/` 폴더 전체, HTML, JSON, PDF, 내부 보고서를 통째로 복사하지 않는다. 실제 반출 범위는 반드시 기관 심사자가 확정한다.

---

## 16. 실행 기록 양식

현장에서 아래 양식을 복사해 채운다.

```text
[방문 정보]
날짜/시간:
실행 담당자:
기관 담당자:

[경로]
패키지 폴더:
데이터 폴더:
결과 폴더:
Python sys.executable:

[환경]
Python 버전:
OS:
GPU/CUDA:
허용 CPU thread 수:
예약 종료 시각:

[단계]
survey-only 시작/종료/결과:
check 시작/종료/결과:
mock 시작/종료/결과:
mock 검증 결과:
full 시작/종료/결과:
full 검증 결과:

[데이터 확인]
annotation/label/EMR 연결:
형식 변종:
파싱 오류:
산모 ID 의미 확인:
0/9999/빈칸 의미 확인:
측정 단위·시각·장비 확인:
판독자·라벨 생성 과정 확인:

[보존]
LATEST_VISIT.json 위치:
full 실행 폴더:
내부 결과 보존 위치:
기관 반출 심사 상태:

[다음 행동]
관측된 문제:
가능한 설명:
다음에 바꿀 한 요소:
필요한 데이터/환경:
판단 기준:
```

---

## 17. 퇴실 전 체크리스트

- [ ] `status.json`의 상태를 직접 확인했다.
- [ ] `tools/validate_run.py` 결과를 보존했다.
- [ ] `LATEST.json`과 `LATEST_VISIT.json`이 가리키는 경로를 확인했다.
- [ ] `visits/`, `mock-.../`, `full-.../`를 결과 루트와 함께 보존했다.
- [ ] `onsite_figures/index.html`, 내부 보고서, 방문 진단이 열린다.
- [ ] 실패가 있었다면 `console.log`, `failure.json`, `events.jsonl`을 보존했다.
- [ ] 데이터 의미에 관한 미확인 질문과 기관 답변을 기록했다.
- [ ] mock 수치를 본 분석 결과로 사용하지 않았다.
- [ ] `internal/` 자료와 `visits/` 전체를 외부로 가져가지 않았다. 방문별 심사 후보는 `visits/<방문ID>/export_review/`에서 구분했다.
- [ ] 반출 후보도 기관 승인을 받기 전에는 복사하지 않았다.

더 자세한 데이터·분석 계약은 [README.md](README.md), 첫 방문의 의사결정 전략은 [FIRST_VISIT_STRATEGY.md](FIRST_VISIT_STRATEGY.md), 산출물과 반출 구분은 [OUTPUTS.md](OUTPUTS.md), 로컬 검증 근거는 [MOCK_TEST.md](MOCK_TEST.md)를 참고한다.
