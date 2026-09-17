# 본선 반입 패키지 · 분석 설계 2.0

**처음 실행한다면 [현장 실행 매뉴얼](FIELD_EXECUTION_RUNBOOK.md)부터 읽는다.** Python 환경을 고르는 법, 처음 보는 데이터 조사, 사전검사 → mock → full 순서, 재개·장애 대응·퇴실 전 보존까지 초보자 기준으로 정리했다.

데이터 경로와 결과 저장 경로만 지정하면 실험 A·B, 부가 분석, 공식 모델 재현, 내부 검토 보고서와 **반출 심사용 집계 묶음**을 생성한다. 원천 데이터, 기존 실험 결과, 자체 사전학습 모델, 가상환경, 도커 이미지는 포함하지 않는다. 런타임은 이 폴더만으로 독립하며 인터넷·다른 실험 폴더·학습된 자체 모델을 요구하지 않는다.

**현장에서 결과를 이해하려면 `export_review/onsite_figures/index.html`부터 연다.** 데이터 구성 → 모델 비교 → 오류·경보 → 인자 해석 → 사이트 → 아웃컴 순서의 질문별 그래프와 해설이다. 혼동행렬, 점수 분포·확률 보정, SHAP 방향, 실제 인자 값 범위별 판독 비율을 함께 보여준다. 각 그림은 PNG와 PDF로 저장하며 새 실행에서 자동 생성한다. 설치된 한글 폰트가 있으면 한글, 없으면 영문으로 그린다. HTML 안내는 한글이다. 집계 CSV는 `export_review/csv/`에 모은다.

이미 끝난 실행에는 `python -B tools/build_onsite_figures.py /결과/full-실행해시`로 **재학습 없이** 추가한다. 새 schema 3 실행에서는 `export_review/`에 생성하고, 과거 실행에는 기존 레이아웃을 보존해 `internal/`에 추가한다. 기존 폴더가 있으면 `--name onsite_figures_v2`처럼 새 이름을 지정한다. 이 자료는 개별 구간 SHAP·원 사이트 코드·실제 값 범위를 포함하는 현장 전용이며, `export_review` 안에 있어도 선별된 반출 후보가 아니다.

첫 방문에서 무엇을 조사하고 두 번째 방문의 실험을 어떻게 결정할지는 [FIRST_VISIT_STRATEGY.md](FIRST_VISIT_STRATEGY.md)에 정리했다. **반출 검토용 방문 진단은 `export_review/visit_audit/index.html`에서 연다.** 원천 구성·연결·결측·기관별 집계, 혼동행렬·보정·고정 임계값 ROC/PR, 집계 SHAP 방향, 모든 학습 후보의 종료 진단·반복 이력·곡선, 실행 시간·자원·다음 방문 준비표를 CSV·PNG·HTML로 생성한다. 개별 신호·예측·식별자·원본 경로·가중치·자유문 로그와 작성한 기관 답변은 `internal/`에 보존한다.

## 첫 방문 조사와 실행 일지

기존 전체 실행 명령을 그대로 쓰면 조사·일지를 함께 남긴다. 현장 입력이 낯설거나 GPU/학습 환경이 준비되지 않았을 때는 **조사만 먼저** 실행할 수 있다. 표준 라이브러리 기반 조사이며 ML 환경 사전검사·학습은 하지 않는다.

```bash
# 저장소 루트에서 로컬 원천 데이터 조사만
FETALGUARD_PYTHON=/home/yabsed/miniconda3/bin/python \
bash 본선/02-반입할-파일/RUN.sh \
  --survey-only \
  --data "$PWD/dataset/korean-ctg/dataset" \
  --output "$PWD/.scratch/first-visit-survey"
```

결과 폴더의 `LATEST_VISIT.json`이 최신 조사 보고서와 일지 위치를 가리킨다. Jupyter에는 **데이터 조사만** 버튼이 있다. 정상 실행에서는 `LATEST.json`의 `visit_audit`도 확인할 수 있다. `--survey-only`와 `--check`는 기존 `LATEST.json` 분석 포인터를 덮어쓰지 않는다.

- 원천 조사: 파일/키/타입/9999·빈칸 분포, ID별 파일 연결, 형식·길이·간격 변종, 보간 전 FHR/TOCO 0·평탄 구간·비정상 수치, 기관 코드별 품질, 서로 다른 ID의 동일 원천 신호 표현. 파싱 실패는 기록하고 다른 파일 조사를 계속한다. 엄격한 학습 입력 계약은 그대로다.
- 실행 일지: 패키지 검사 전부터 `visits/<방문ID>/internal/visit_audit/`에 `events.jsonl`, `console.log`, 15초 주기의 `resources.jsonl`을 저장한다. 사전검사 실패에도 남는다. Python stdout/stderr를 보존하며 일부 native 라이브러리 직접 출력은 빠질 수 있다. CPU는 프로세스 누적/구간 사용량, RAM은 지원되는 OS에서 프로세스 RSS, GPU/VRAM은 **장치 전체** 관측이다. 지원하지 않는 계측은 미수집으로 남긴다.
- 학습 진단: CNN은 매 epoch의 loss·validation AP·실제 시작 학습률·시간을 저장한다. CatBoost/XGBoost는 완료한 fit의 실제 반복 이력과 최고 반복·상한·patience를 저장한다. 최고 모델에 남은 트리 개수를 총 학습 반복 수로 간주하지 않는다. fit 도중 강제 종료한 트리의 반복 이력은 복구할 수 없으며 시작/실패 이벤트를 확인한다.
- 준비표: 관측·근거·가능한 설명·다음 비교·바꿀 한 요소·필수 입력·비용의 측정 여부·판단 기준을 `next_visit_plan.csv`에 남긴다. `questions.md`에는 기관 답변을 기록하며 보고서를 다시 생성해도 덮어쓰지 않는다.

원천 통계는 **전체 파일/원천 구간** 기준이므로 중복 배포·두 태아가 포함될 수 있다. 최종 학습의 선택 태아·산모·제외 후 구간과 분모가 다르다. 파일 연결은 분석 적격 판정이 아니다. 출생일을 측정 시각으로 대용하지 않으며 라벨/단위/0의 의미는 기관에 확인한다.

Ctrl+C 및 지원되는 환경의 SIGTERM에서는 실패/중단 요약과 반출용 진단을 생성한다. 전원 차단·SIGKILL에서는 이미 저장한 일지와 마지막 체크포인트만 남고 최종 상태가 `running`으로 남을 수 있다. 결과 루트 전체(특히 `visits/`)를 내부에 보존해야 상세 진단 근거가 유지된다. 반출용 진단은 내부 링크 없이 독립적으로 열린다. 재개 시 `visit_audit_002`처럼 새 스냅샷을 만들며 `LATEST_VISIT.json`이 최신 반출용 화면을 가리킨다. 조사만 실행한 경우에도 `visits/<방문ID>/export_review/visit_audit/`에 CSV·HTML을 생성한다.

기존 실행의 이력으로 **재학습 없이** 진단만 만들 수도 있다. 원천 조사를 다시 수행하지 않으며 과거에 기록하지 않은 트리 이력·자원 로그는 미수집으로 표시한다.

```bash
python -B tools/build_export_diagnostics.py /결과/full-실행해시
# 새 진단 스냅샷을 추가하려면
python -B tools/build_export_diagnostics.py /결과/full-실행해시 --name visit_audit_v2
```

`tools/build_visit_audit.py`도 기본 출력이 `export_review/visit_audit/`다. 기존 심사 묶음·학습 결과의 해시는 보존하고 추가 집계는 자체 `EXPORT_MANIFEST.json`으로 검증한다. 같은 출력 이름의 완성본은 검증 후 재사용한다. `--output`을 명시한 이전 방식은 상세 내부용 보고서 생성용이다.

산모 수 learning curve와 추가 예산 학습은 **아직 자동 실행하지 않는다**. 데이터 양 포화는 미측정으로 표시한다. 이번 변경은 실험 예산·분할·모델 선택 규칙을 바꾸지 않으며 test 성능으로 연장 후보를 정하지 않는다.

**이미지 반출 심사 후보:** 기존 성능 집계는 **`export_review/images/`**, 추가 진단·학습곡선은 **`export_review/visit_audit/images/`**의 PNG다. CSV·HTML·JSON은 각각의 상위 검토 폴더에 둔다. CSV 반출 검토가 가능하면 `export_review/csv/`와 `export_review/visit_audit/csv/`를 사용한다. 상세 구성은 [OUTPUTS.md](OUTPUTS.md)에 있다.

기존 분석을 재학습하지 않고 그래프를 추가하려면 `python -B tools/build_image_review.py /결과/full-실행해시`를 실행한다. 별도로 생성된 **`image_review/images/`**가 이미지 전용 심사 폴더다.

로컬 파일럿 결과를 반영해 단일 인자 비선형 대조군, Cat18, 평활 민감도, 동일 조건 CNN/EMR 비교를 추가했다. 질문과 비교·해석 범위는 [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md), 현장/반출 산출물은 [OUTPUTS.md](OUTPUTS.md), 실행 검증은 [MOCK_TEST.md](MOCK_TEST.md)를 읽는다. 파일럿을 새 가설의 독립 확증 결과로 취급하지 않는다.

## 현장에서 실행

**JupyterLab:** 신청한 PyTorch / Python 3.10 이상 커널로 `START.ipynb`를 열고 **Run All** → 데이터 폴더·결과 폴더 입력 → **전체 실행 / 이어서 실행**. 코드 수정은 필요 없다. 먼저 **사전검사** 버튼으로 환경과 공식 모델 로딩을 확인할 수 있다.

이미 분석한 산모 목록이 있으면 선택 입력창 `기존 산모 CSV` 또는 `--prior-cohort /내부/기존산모.csv`를 사용한다. `mother_id` 열이 필요하며 같은 산모 ID 체계여야 한다. 해당 산모는 주 holdout의 train에만 포함하고 validation/test에서는 제외한다. CV·H5·LOSO는 기존 산모를 포함하는 탐색 분석이다. 미제공은 `not_checked`로 기록한다. 목록은 반입 폴더에 동봉하지 않는다. `tools/prepare_prior_cohort.py 기존실행 --output /내부/기존산모.csv`로 과거 실행에서 내부용 목록을 만들 수 있다.

**터미널:** 아래 한 명령으로 실행한다. 경로 인자를 생략하면 터미널에서 경로를 물어본다.

```bash
bash RUN.sh --data /안심존/태아심박동데이터 --output /팀폴더/분석결과
```

GPU가 있으면 자동으로 사용하고 없으면 CPU로 실행한다. CPU 강제는 `--device cpu`. Windows에서는 `RUN.cmd` 또는 `python -B run.py`를 사용한다. Jupyter 커널과 터미널 Python은 다를 수 있으므로 `FETALGUARD_PYTHON=/경로/python bash RUN.sh ...`로 명시할 수도 있다.

Python 패키지는 기존 사전 신청 목록을 사용한다. `requirements-approved.txt`는 신청 목록 사본, `requirements-runtime.txt`는 실제 실행에 필요한 부분집합이다. 런처는 패키지를 설치하거나 모델을 다운로드하지 않는다. 공식 YOLO는 동봉한 TorchScript 변환본을 사용하므로 실행 시 torchvision이나 오래된 YOLO의 부가 의존성을 불러오지 않는다.

## 입력 데이터 계약

압축이 풀린 데이터 상위 폴더를 지정한다. `Training/Validation/Test`로 나뉘어 있어도 재귀 검색한다. 로컬의 분류·탐지·혼합본을 한꺼번에 지정할 수 있다.

| 파일 위치의 마지막 폴더 | 용도 | 요구 조건 |
|---|---|---|
| `annotation_person/*.json` | FHR/TOCO 수치 신호 | 5분·150표본·2초 간격, twins/code/data |
| `labels/*.json` | 세그먼트 Abnormality, FIGO 주석, Emergency, Bbox | 파일명과 ID 일치, 0/1 문자열의 길이와 선택 태아 구간 수 일치 |
| `emr/*.json` | 산모 식별·부가 분석 | `Mother.de-identification_ID` 필수; 결측 산모 ID를 레코드 ID로 대체하지 않음 |
| `refine_images/*.png` | Bbox 좌표 전수 검증·공식 YOLO 평가 | 있으면 자동 사용, 없으면 검증/평가 불가 범위 명시 |

레코드 수는 제한하거나 고정하지 않는다. 2,301건/2만 건에 같은 경로를 사용한다. 동일 ID의 내용이 같은 중복 파일은 제거하고, 다른 신호·라벨을 가진 중복 ID는 오류로 중단한다. 분류용 라벨의 Bbox가 비어 있고 나머지 필드가 같으면 탐지용 라벨의 Bbox를 보충한다. 신호가 없는 탐지 전용 레코드는 공식 YOLO 재현에만 사용하고 주 분석에서 제외한다.

다태아는 ID의 `_1`/`_2`와 `twin1`/`twin2`를 대응시킨다. FHR 결측률이 30%를 초과한 구간은 제외한다. 표본 간격/길이, 라벨 정렬, 산모 ID가 계약과 다르면 오류를 기록하며 임의로 맞추지 않는다. 입력 계약이 현장 데이터와 다를 때까지 무조건 실행을 보장하는 범용 파서는 아니다. 회신안에서 문의한 수치 JSON·기관 접두어·산모 ID 보존이 필요한 이유다.

## 보고서와 실행 코드의 대응

| 보고서 내용 | 구현 | 주요 결과 |
|---|---|---|
| 2.3 분포·결측 | `fg/data.py` | 기관·산모·다태아·세그먼트 수, EMR 결측표, 제외 사유 |
| 2.4 검증 질문 1 | `fg/data.py` | `bbox_audit.csv`: 실제 PNG 크기/경계/높이/이상 위치 전수 검증 |
| 4.1·5.1 인자 28개 | `fg/features.py`, `feature_rules.py` | `segments.csv`, 신호 캐시 |
| H1 실험 A | `fg/models.py` | Cat28·XGB28·Cat18·전체 인자 로지스틱, 산모 단위 반복 CV와 홀드아웃 |
| H2 단일 인자와 조합 | `fg/models.py` | 선형 28개 + spline 비선형 28개, 검증 AUROC로 선택 후 조합과 비교 |
| H3 기여 구조 | `fg/models.py` | 6그룹 제거·robust, 평활 유무, seed별 차이, 훈련 인자-판독 관계·SHAP |
| H4 실험 B | `fg/cnn.py` | seed 평균 검증 AP로 용량 선택, 같은 width/seed 정규화 절제, CI·추론 비용 |
| 5.3.1 H5 | `fg/supplementary.py` | pH·Apgar 대응, 임상 변수에 판독 요약을 추가한 OOF 증분 |
| 5.3.2 주석 대조 | `fg/supplementary.py` | 기록 단위 기저선·변이도·감속/가속과 추출 인자 대조 |
| 5.3.3 기관 이질성 | `fg/supplementary.py` | 기관별 성능, 전 기관 LOSO, source 검증셋만으로 Platt 보정 |
| 5.3.4 EMR 추가 | `fg/supplementary.py` | 동일 seed 후보·예산·선택 규칙의 파형 vs 파형+EMR, 하위군 |
| 공식 모델 재현 | `fg/official.py` | XGBoost Emergency 별도 평가, YOLO 공통 이미지 부분집합 비교 |
| 측정 타당도 | `fg/data.py` | 특성별 0 비율·상수·측정 불가, 무평활 민감도, 연속 구간 경계 |
| 결과·반출 심사 | `fg/report.py`, `fg/export_review.py` | 현장 보고서와 식별자·가중치 없는 고정 구조 집계 묶음 |

보고서 5.3은 저장소의 `experiment-14/2-proposal-draft/draft_3_to_6.md`에 남은 분석 계획까지 포함했다. 최신 인자 정의는 `experiment-15`의 A.2 정렬 구현을 기준으로 옮겼다. 출처 파일과 원본 SHA256은 `SOURCE_MANIFEST.json`에 있다.

## 분석 규약

- 산모 그룹을 보존한 약 80/10/10 train/val/test 분할. 원 배포 폴더와 무관하게 새 분할을 만들며 동일 산모의 쌍둥이·반복 기록을 묶는다. 분할표는 결과에 저장한다.
- 실험 A의 반복 CV는 최종 test를 제외한 개발 코호트에서 수행한다. 각 fold 안에 그룹 검증셋을 두어 조기중단·운영 임계값을 결정한다. OOF 파일은 시드별 wide 형식으로 저장해 대규모 자료에서 중복 행의 메모리 사용을 줄인다.
- CatBoost/XGBoost는 **validation AP**로 seed를 선택한다. CNN은 seed별 validation AP 평균으로 width를 선택하고 해당 width의 최고 validation AP seed를 선택한다. H2는 주 지표·선택 기준을 **AUROC**로 맞춘다. H1/H4 주 지표는 AP이며, 테스트 성능으로 모델을 선택하지 않는다.
- 특이도 90%의 운영 임계값은 validation의 음성 점수에서 정한다. 동점 때문에 정확히 90%를 만들 수 없으면 보수적으로 잡는다. `sensitivity_at_test_spec90`은 별도의 test ROC 기술통계이며 운영 임계값 성능과 구분한다.
- AUROC/AUPRC, 민감도·특이도·PPV·F1, Brier, 완전 정상 기록 경보율, 정상 기록의 관측 시간당 양성 5분 창 수를 산출한다. CI와 모델 차이는 산모 클러스터 부트스트랩으로 계산한다. 연속 양성 창을 하나의 경보 이벤트로 합치지는 않는다.
- CNN 입력은 보고서대로 **2×150 원시신호**이며 종전 실험 8의 통계량 7개 추가 입력은 붙이지 않는다. 주 정규화는 training에서 구한 채널별 최대절댓값 나눗셈으로 심박수 수준을 보존한다. 세그먼트별 z-정규화는 별도 절제 대조군이다.
- 0.5Hz에서 15초 최소 지속은 8표본(16초)로 올림한다. 지연 상관 탐색은 관측 가능한 2초 간격이다. `stv`는 표본 간 차이이며 박동 간 임상 STV가 아니다. 5분 창에서 장기/중증 감속과 10분 기저선에 한계가 있다.
- 기본 인자는 기존 A.2 구현대로 FHR/TOCO에 30초 평활을 적용한다. TOCO=0을 결측으로 간주하는 기존 인자 전처리 관례도 유지했다. CNN은 FHR의 0만 보간하며 TOCO의 0·음수를 보존한다. TOCO 스케일/결측 의미는 데이터 인벤토리와 함께 해석해야 한다.
- 중요도는 예측 기여이며 전문의 사고의 인과 설명이 아니다. 차이의 CI가 0을 포함하는 것만으로 H4의 동등성을 선언하지 않는다. 자동 보고서는 가설 채택을 자동 선언하지 않는다.
- H5의 pH/Apgar 분석은 관측 아웃컴이 있는 기록에만 적용한다. 기록 단위 FIGO 주석을 개별 세그먼트의 정답으로 복제하지 않는다. 복수 판독자가 제공되지 않으면 판독자 간 신뢰도를 만들어내지 않는다.
- LOSO의 calibration은 제외 기관의 라벨을 보지 않고 source validation에서만 학습한다. 다른 기관에 같은 산모가 있으면 해당 산모도 학습에서 제외한다. EMR 추가 실험에는 출생 후 결과·분만방법·Emergency가 들어가지 않는다. 배포 EMR의 재태주수는 분만 시점 값이므로 모니터링 시점 가용성은 별도 확인 대상이다.

## 전체 모드와 mock

| 항목 | full (기본) | mock |
|---|---|---|
| 읽는 원천 데이터 | 전량 | **전량** |
| 실험 A CV | 5-fold × 3 seeds | 2-fold × 1 seed |
| 트리 상한 / patience | 1,000 / 80 | 30 / 10 |
| CNN | 5 widths × 3 seeds × 2 norms = 30회 | 2 widths × 1 seed × 2 norms = 4회 |
| CNN epoch 상한 / patience | 1,000 / 150 | 3 / 2 |
| 부트스트랩 | 2,000회 | 50회 |
| 부가 분석·공식 모델·그림 | 모두 | 모두 |

mock은 연결과 실행 가능성 검사다. 성능·CI를 연구 결론으로 사용하지 않는다. H5/LOSO는 full에서도 고정 시드 하나를 사용하며 H5는 해당 모드의 CV fold 수를 따른다. EMR 유무는 해당 모드의 모든 seed 후보를 동일하게 사용한다. full은 GPU에서의 긴 실행을 전제로 하고 CPU에서는 훨씬 오래 걸릴 수 있다. `hit_cap`은 충분한 수렴을 확인하지 못한 실행을 표시한다.

현재 저장소에서는 다음 한 명령으로 로컬 데이터 전체 mock을 실행한다.

```bash
bash 본선/02-반입할-파일/MOCK.sh
```

이 편의 스크립트만 원래 저장소의 데이터 위치를 기본값으로 사용한다. 패키지만 옮긴 환경에서는 `RUN.sh --profile mock --data ... --output ...` 또는 노트북을 쓴다. mock 결과 기본 위치는 저장소의 `.scratch/import-mock-results/`이며 반입 폴더에 들어가지 않는다.

## 포함한 외부 모델

1. **AI-Hub 71366 태아상태진단 분류모델 v1.0 / XGBoost.** 원본 `xgb_clf.model`과 XGBoost 1.6.2에서 내보낸 JSON 변환본. 타깃은 기록 단위 `Emergency`이며 44개 입력에 전문의 판독 비율과 사후 정보가 포함된다. 주 연구의 파형→Abnormality 모델과 같은 과제로 취급하지 않는다. 공식 파서 오류까지 재현한 팔과 쌍둥이·부호·문자열 경계를 고친 팔을 모두 평가한다. 후자는 같은 가중치의 입력 교정 민감도 분석이다.
2. **AI-Hub 71366 태아이상시점 탐지모델 v1.0 / YOLOv5s.** 원본 `best.pt`와 변경 없이 추론 형식으로 내보낸 `inference.torchscript`. 원본 float 추론과 변환본의 무작위 입력 출력 차이를 확인했다. 고정 640×640 square letterbox를 사용하므로 종전 val.py의 rectangular batching mAP를 그대로 재현한다고 주장하지 않는다. 확률은 박스 중심을 해당 세그먼트에 대응시켜 최대 confidence로 만든다. 이미지 없는 기록은 0점으로 채우지 않는다.

원 배포 소스는 `third_party/`, 동봉 라이선스는 `third_party/LICENSE`, 모델 설명은 `third_party/AIHUB_README.md`에 있다. 모델 제공자는 AI-Hub 구축기관이며 YOLOv5 기반 구현은 Ultralytics다. CTG-net/Chiou 비교군은 저장소에 있던 **논문 재구현**을 사용해 현장 데이터로 처음부터 학습한다. Google이 배포한 학습 가중치라고 표시하지 않는다. 공식 가중치의 학습 코호트 중복 여부가 불명확하므로 공식 모델 성능은 배포 모델 재현/참조 결과다.

## 결과·중단·재개

결과는 지정한 폴더 아래 `full-<실행해시>` 또는 `mock-<실행해시>`에 저장된다. 현장 시작 화면은 **`export_review/onsite_figures/index.html`**, 세부 보고서는 **`internal/report/report.html`**, 집계 검토 화면은 **`export_review/report.html`**, 이미지 제출 후보는 **`export_review/images/`**다. 집계 CSV는 **`export_review/csv/`**다. 상위 `LATEST.json`에 화면 경로가 기록된다.

`internal/` 아래 `data/`, `splits/`, `experiment_a/`, `experiment_b/`, `supplementary/`, `official/`, `report/`가 순서대로 만들어진다. 다음으로 `export_review/` 집계 묶음과 그 안의 `onsite_figures/`를 생성한다. 원본·모델·개인별 예측은 `internal/`에 보관하고, 현장 그림의 입력 해시는 `onsite_figures/manifest.json`에도 기록한다. 실행 전에 `protocol.json`으로 규약을 기록한다. 환경, 설정, 입력 파일 해시, 패키지 해시, 기존 산모 목록 해시가 실행 ID에 들어가므로 서로 다른 데이터·버전·예산의 결과가 섞이지 않는다.

같은 명령을 재실행하면 완료 단계는 산출물 해시를 검증한 뒤 건너뛴다. 중간 학습 단계에서는 이미 저장한 트리, CNN epoch 체크포인트, 이미지별 YOLO 추론 결과를 재사용한다. 완료 산출물이 훼손되면 덮어쓰지 않고 오류로 알린다. 같은 출력에 동시 실행하는 것은 OS 파일 잠금으로 막는다. 오류는 비정상 exit code와 `status.json`에 남는다.

`config.json`만 현장 설정 파일로 수정할 수 있다. 다른 승인 파일이 바뀌면 무결성 검사가 실패한다. 일부 분석을 의도적으로 비활성화할 경우 `cnn` / `official_models`를 false로 설정할 수 있지만 결과에 미수행을 명시한다. 누락 아웃컴·단일 클래스 기관 등은 상태 파일에 이유가 남는다. 수치 신호·산모 ID·주 라벨의 오류는 전체 분석을 중단한다.

`export_review/`의 집계 보고서는 허용한 집계 필드로 CSV·HTML·그림·설정 요약을 생성한다. 이미지 심사 후보는 `images/`와 `visit_audit/images/`, CSV 심사 후보는 `csv/`와 `visit_audit/csv/`다. 선별 집계에는 식별자, 원본 경로·파일별 해시, 개인별 예측, 가중치를 포함하지 않는다. 작은 집단은 기본 `export_min_mothers=10`으로 선별·억제한다. 루트 `EXPORT_MANIFEST.json`은 기존 성능 집계, `visit_audit/EXPORT_MANIFEST.json`은 추가 진단의 목록·해시를 관리하며 검증기가 둘 다 검사한다. `onsite_figures/`는 비선별 현장 전용 자료로 자체 manifest와 별도 완료 마커를 사용한다. 기존 결과는 보존한다.

## 개발·검증·반입 ZIP

```bash
python -B -m unittest discover -s tests -v
python -B tools/validate_run.py /결과/mock-실행해시 --expect-records 2301 --check-original-xgb
python -B tools/seal_package.py --zip ../02-반입할-파일-v2.zip
```

`PACKAGE_MANIFEST.json`은 설정 파일을 제외한 승인 파일의 SHA256이다. 반입용 ZIP 생성기는 심볼릭 링크와 데이터/파생 테이블을 거부한다. 원본 자산 갱신용 `tools/prepare_assets.py`, 원래 환경에서 공식 가중치 변환을 재현하는 `tools/export_official.py`는 유지보수용이다. 현장에서 실행할 필요는 없다.
