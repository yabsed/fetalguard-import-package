# 실행 후 확인·보존·반출 심사

현장에서 사람이 결과를 읽을 때는 **`export_review/onsite_figures/index.html`**부터 연다. 질문별로 관련 지표를 묶은 그래프와 해설이 있으며, PNG/PDF를 직접 열 수도 있다. 세부 수치·개별 사례는 연결된 기존 보고서에서 확인한다. 새 실행의 생성 구조이며 기존 결과 디렉터리를 옮기지 않는다.

이미지 형태의 그래프만 허용되는 경우 **`export_review/images/`의 PNG만 심사에 제출한다.** CSV를 제출하거나 표 전체를 스크린샷으로 바꿀 필요가 없다. 이 폴더에는 선별된 집계값을 새로 그린 그래프만 들어 있다. 실제 허용 범위는 기관 심사로 확정한다.

파일 확장자와 함께 **내용과 생성 경로**를 확인한다. 개별 파형·개인별 예측을 이미지로 저장했다고 반출 가능한 것은 아니다.

```text
결과/full-실행해시/
├── status.json
├── internal/                    # 현장 전용
│   ├── protocol.json · run_manifest.json
│   ├── data/ · splits/
│   ├── experiment_a/ · experiment_b/
│   ├── supplementary/ · official/
│   └── report/report.html · case_review.html · case_review.csv · case_*.png
└── export_review/               # 현장 검토·심사 준비 자료
    ├── onsite_figures/          # 비선별 현장 전용: 반출 후보가 아님
    │   └── index.html · *.png · *.pdf · README.md · manifest.json
    ├── csv/                     # 선별 집계 CSV 전체
    │   └── model_comparison.csv · feature_response_train.csv · ...
    ├── images/                  # 이미지 전용 제출 후보: PNG만
    │   ├── 00_coverage_p*.png · 00_protocol_p*.png
    │   └── 01_model_comparison_p*.png · ...
    ├── report.html · report.md
    ├── figures/
    ├── protocol_summary.json
    └── EXPORT_MANIFEST.json
```

`export_review/`는 현장용과 심사용을 함께 열어보는 작업 폴더다. 아래 비교의 오른쪽은 `onsite_figures/`를 제외한 선별 집계에만 해당한다. 현장 그림에는 개별 SHAP·실제 값 범위·원 사이트 코드가 있다.

| 구분 | 내부 전용 `internal/` | 선별 집계 (`export_review/onsite_figures/` 제외) |
|---|---|---|
| 보고서 | 전체 분석, 개별 설명 파일·내부 산출물 링크 | 집계 수치에서 새로 작성한 HTML·Markdown |
| 데이터 | 신호 캐시, 개인별 인자·EMR·라벨·분할·예측 | 충분한 집단의 표본 수·성능·CI |
| 설명 | 개인별 SHAP와 값, 정탐·오탐·미탐 파형 사례 | 평균 중요도, 집계 인자-판독 관계, 그룹 제거 차이 |
| 사이트 | 원 사이트 코드와 개별 자료 | S01 등 별칭과 선별된 집계 |
| 재현 정보 | 경로·파일명·원본별 해시·환경·전체 설정 | 허용한 설정·코드 패키지 해시·묶음 파일 해시 |
| 모델 | 가중치·학습 상태·체크포인트 | 모델 종류·선택 설정·비용 요약만 |

## 현장에서 할 일

1. `status.json`이 `complete`인지 확인하고 `export_review/onsite_figures/index.html`을 연다. 세부 표는 `internal/report/report.html`에서 확인한다.
2. 인자 측정 가능성, 분석 누락, 학습 상한 도달, H1–H5의 비교 CI, 사이트별 오경보, 아웃컴 결측을 확인한다.
3. `internal/report/case_review.html`에서 Cat28의 TP/TN/FP/FN별 최대 2개 파형을 확인한다. 미리 정한 극단 점수 사례이며 대표 표본이 아니다. 개별 SHAP가 계산되지 않은 사례는 미산출로 표시한다. 개인별 설명·예측·모델은 내부에서 검토하고, ID를 지웠다는 이유만으로 이런 행을 반출 묶음에 추가하지 않는다.
4. `tools/validate_run.py`로 단계별 해시와 심사 묶음의 무결성을 검증한다.
5. `export_review/report.html`의 PNG 그래프 묶음과 현장용 집계표를 대조한다. 이미지 전용 심사에는 **`images/` 안의 PNG만** 선택한다. HTML·CSV·JSON·PDF가 들어 있는 상위 폴더 전체를 함께 제출하지 않는다. 승인된 이미지 파일만 시설 절차대로 반출한다.

`export_review`는 승인 여부를 뜻하지 않는다. `EXPORT_MANIFEST.json`의 상태는 `pending_institution_review`다. 자동 외부 전송 기능은 없다. 기관이 모델·개별 사례 등 추가 산출물의 반출을 승인한다면 별도 범위로 처리하며, 기본 묶음에는 포함하지 않는다.

## 현장 이해용 그래프

`export_review/onsite_figures/`는 사람의 질문 순서로 읽는 현장 전용 자료다. 전체 결과를 지표마다 분할하는 대신 비교에 필요한 결과를 같은 그림에 배치한다.

| 질문 | 그래프 |
|---|---|
| 어떤 자료로 학습하고 평가했나? | 분할별 정상/이상 구간 수·산모 수·양성 비율 |
| 어떤 모델이 낫고 어떤 변경이 도움이 되나? | 주요 모델 AP/AUROC·CI, 조합/인자 확장/평활 제거/EMR의 짝지은 차이 |
| 무엇을 놓치고 얼마나 경보하나? | 건수와 행 비율을 표시한 혼동행렬, 분모별 탐지율·정밀도·정상 기록 경보율 |
| 점수와 확률을 믿을 수 있나? | 실제 클래스별 점수 분포, 저장된 검증 임계값, bin 표본 수가 있는 보정 곡선, ROC/PR 운영점 |
| 무엇이 예측에 기여하나? | 평균 SHAP와 인자군 제거 비교, 인자 값의 상대 순위로 색칠한 개별 SHAP 점 |
| 인자 값과 판독이 어떻게 연결되나? | SHAP 상위 8개 인자의 훈련 bin 반응 곡선, 실제 값 범위와 구간 수 |
| 인자 측정은 제대로 됐나? | 전체 인자 0 비율, 상수·측정 불가 항목 구분 |
| 사이트가 바뀌면 어떠한가? | 원 사이트별 양성 비율·특이도·정상 기록 경보율·CI, 보정 전후 Brier와 경보율 |
| 판독과 아웃컴은 어떻게 연결되나? | 관측 기록의 사건 비율·분모·결측 규모, 판독 요약 추가의 OOF 성능 차이 |

이번 구성이 모두 가능한 실행에서는 15개 그림을 PNG/PDF 각각 저장한다. 수행되지 않은 CNN의 점수는 추가하지 않는다. 사이트/SHAP 등 자료가 없으면 해당 그림은 생략한다. `mock`과 실제 실행 예산을 표시하며, 테스트셋으로 모델·임계값을 다시 고르지 않는다. 인자 반응은 훈련 데이터만 사용한다. 한글 폰트를 자동 탐색하며 없으면 그림을 영문으로 생성한다. 폰트나 패키지를 다운로드하지 않는다.

기존 완료 실행에서도 다음과 같이 추가한다.

```bash
python -B tools/build_onsite_figures.py /결과/full-실행해시
# 이미 같은 폴더가 있으면 새 이름으로 생성
python -B tools/build_onsite_figures.py /결과/full-실행해시 --name onsite_figures_v2
```

새 schema 3 실행의 기본 출력은 **`full-실행해시/export_review/onsite_figures/`**다. 이전 schema 1/2 실행을 후처리할 때는 원래 심사 묶음과 완료 해시를 보존하도록 `internal/onsite_figures/`에 추가한다. 읽은 입력과 새 산출물의 해시는 자체 `manifest.json`에 기록한다. 새 파이프라인에서는 집계 생성 후 별도 단계로 생성하며, 실행 루트의 `.state/onsite_figures.json`으로 재개 검증하고 `LATEST.json`에 시작 페이지를 기록한다. 개별 구간 SHAP·실제 범위·원 사이트 코드가 포함되므로 현장 전용으로 보존한다.

## 이미지 전용 그래프 묶음

PNG는 가로 **4,000픽셀 / 200dpi**, RGB로 저장한다. 영문 그래프 표기는 외부 폰트 설치 없이 오프라인 환경에서도 깨지지 않도록 한다. 한 그림에 최대 6개 비교 행·2개 지표를 배치하고, 후보나 지표가 더 많으면 페이지를 추가한다. 인자-판독 곡선은 페이지당 최대 4개 인자이며 가능한 모든 인자를 포함한다. PNG만 열어도 제목·분석 범위·수치·단위·표본 규모·CI·해석 제한을 확인할 수 있게 구성한다.

| 이미지 묶음 | 포함하는 결과 |
|---|---|
| `00_coverage`, `00_protocol` | 전체 28개 집계 영역의 수행/억제 상태, 분석 예산·seed·코드 패키지 해시 |
| `model_comparison`, `paired_comparisons`, `cohort_counts` | 주 성능·차이·산모 군집 95% CI, 분할별 표본 규모 |
| `cv_summary`, `validation_selection`, `single_feature_selection`, `seed_*` | 반복 CV, 선형/비선형 단일 인자, 검증셋 선택과 seed별 변동 |
| `feature_importance`, `feature_response_train`, `feature_quality`, `preprocessing_sensitivity`, `figo_agreement` | 평균 SHAP, 모든 가용 인자의 훈련 구간별 반응, 품질·평활·기록 단위 주석 대응 |
| `cnn_*`, `normalization_matched`, `inference_cost` | 용량·정규화·seed 선택, 같은 조건의 성능 차이, 측정 범위를 구별한 추론 비용 |
| `outcomes`, `outcome_*` | 판독-아웃컴 연관성·예측 증분·교차 집계, 아웃컴 결측과 관측 집단 특성 |
| `emr_*`, `loso`, `official_reference`, `hypothesis_evidence` | EMR 추가, 기관 제외 평가, 별도 과제의 공식 모델 참조, 질문별 근거 |

- 점 옆에 추정값과 `[95% CI]`를 함께 표시한다. 수치는 통상 소수점 4자리이며, 작은 수는 과학 표기법을 사용한다. CI가 없으면 만들어내지 않는다.
- `LEFT - RIGHT`는 제목/행 이름으로 방향을 표시한다. AP/AUROC는 양의 차이가 LEFT에 유리하고 Brier는 음의 차이가 LEFT에 유리하다.
- `SUPPRESSED`는 해당 행의 선별 억제다. `NA`는 미산출 또는 지표별 분모 부족으로 비공개인 값이다. 어느 쪽도 0점/0건으로 그리지 않는다. `N`은 관측 수, `M`은 산모 수이며 정상 경보 지표에는 별도 분모 선별이 적용된다.
- 단위가 다른 인자의 평균·표준편차·평활 변화량은 인자별 축으로 나눈다. 공식 Emergency 모델과 주 과제, 이미지 부분집합도 같은 비교축으로 섞지 않는다.
- 결과가 없는 영역은 coverage 그림에 남는다. `mock`의 모든 그림에 `MOCK: EXECUTION TEST ONLY`를 표시한다.
- 산모별 점·개별 파형·개별 SHAP·원시 CSV의 이미지화는 포함하지 않는다. PNG에 CSV나 원시 배열을 메타데이터·첨부·숨은 문자열로 넣지 않는다. 이미지 파일명, PNG 형식, 해시와 메타데이터 부재를 검증한다.

## 이미 분석을 마친 경우: 재학습 없이 이미지 생성

새 패키지에서 아래 명령을 실행한다. 기존 `internal/`을 읽고 별도의 `image_review/`를 생성하므로 기존 실행의 완료 해시·산출물을 수정하지 않는다.

```bash
python -B tools/build_image_review.py /팀폴더/분석결과/full-실행해시
```

제출 후보는 **`full-실행해시/image_review/images/`**다. `image_review/report.html`, `image_review/csv/`와 `EXPORT_MANIFEST.json`은 현장 검토용으로 남는다. 이 도구는 선별 집계 묶음만 만들며 현장 그림은 별도 도구로 추가한다. 이미 같은 이름의 폴더가 있으면 새 `--output /팀폴더/새심사폴더`를 지정한다. 기관 기준이 더 높으면 `--min-mothers 20`처럼 **기존 기준을 상향**할 수 있다. 이 명령은 기준을 낮추거나 기존 심사 묶음을 덮어쓰지 않는다.

## 자동 선별의 범위

선별 집계는 고정된 파일명·필드 구조로 생성한다. 원래 report HTML, `run_manifest.json`, 임의 JSON, 원시 파일·개인 ID·원본 파일명·데이터별 해시·가중치를 복사하지 않는다. 이 약속은 비선별 `onsite_figures/`에는 해당하지 않는다. 사이트 별칭도 익명성을 보증하지 않으므로 심사 대상이다.

`export_min_mothers`의 기본값은 10이다. 산모 수와 필요한 양성/음성 산모 분모가 부족하거나 확인되지 않으면 해당 집계값을 억제한다. 전체 내부 분석은 그대로 남는다. 다른 표의 합계·구성·희귀 사건을 조합할 때의 노출 가능성까지 기관의 검토가 필요하며, 이 선별을 법적·기관 차원의 익명화 판정으로 간주하지 않는다.

SHAP 평균 중요도는 전체 테스트셋이 아니라 **실제 SHAP 계산 표본**의 산모 수로 선별한다. 아웃컴 관측/결측 집단의 나이·주수 등의 요약도 해당 변수가 실제 관측된 산모 수를 사용한다.

CV·seed별 보조표의 정상 기록 경보율·정상 관찰시간당 양성 창 수는 해당 행의 경보/무경보 산모 분모를 확인하지 않으므로 반출 묶음에서 공란으로 둔다. 이 지표는 별도로 분모를 심사한 주 비교표와 LOSO 표에서 확인한다. 현장 전용 표에는 원래 값이 남는다.

집계만으로 외부에서 임의의 새 하위군 분석·임계값 재선정·개인별 오류 조사를 할 수는 없다. 핵심 분석과 필요한 집계는 퇴실 전에 검토하고 내부 결과의 보존은 시설 정책에 따른다.

## 현장 검토용 집계 원본과 이미지의 대응

아래 CSV는 모두 새 실행의 **`export_review/csv/`** 아래에 저장된다. 이전 schema 1/2의 루트 CSV도 검증기가 계속 지원한다.

- `model_comparison.csv`, `paired_comparisons.csv`: 동일 테스트 코호트 성능·차이·신뢰구간.
- `cv_summary.csv`, `seed_holdout_metrics.csv`, `seed_paired_cat28_minus_comparator.csv`: 반복·seed별 변동과 절제 방향.
- `feature_quality.csv`, `feature_response_train.csv`, `feature_importance.csv`: 측정 제약과 집계 설명.
- `cnn_capacity_validation.csv`, `cnn_capacity_summary.csv`, `cnn_selection.csv`, `normalization_matched.csv`: CNN 선택과 같은 width/seed 정규화 비교.
- `inference_cost.csv`: 모델별 추론 비용. 인자 추출·배치·IO 포함 범위와 함께 읽는다.
- `outcomes.csv`, `outcome_associations.csv`, `outcome_reading_cells.csv`, `outcome_missingness.csv`, `outcome_cohort_descriptors.csv`: 실제 전문의 판독-아웃컴 대응과 증분·결측 집단 특성.
- `emr_selection.csv`, `emr_matched_seeds.csv`, `loso.csv`, `official_reference.csv`: EMR 모델 선택·짝지은 비교, 사이트별 운영 성능, 별도 타깃·부분집합 참조.
- `hypothesis_evidence.csv`, `cohort_counts.csv`, `protocol_summary.json`: 질문별 근거와 분석 규모·설정 요약.
- `EXPORT_MANIFEST.json`: 선별 집계 파일 목록·SHA256·검토 대기 상태. `onsite_figures/`는 목록에서 제외하고 자체 manifest로 검증하며 선별 통과로 취급하지 않는다.

수행 불가 또는 선별 억제 항목은 빈 표/상태로 남을 수 있다. 현장에서 HTML을 볼 때는 검토 폴더 전체를 보존하고, 이미지 전용 제출에는 `images/`의 PNG만 선택한다. 표의 값은 이미지 생성의 현장 대조 자료이며 CSV 반출을 전제하지 않는다. 임의 파일·PNG 메타데이터를 추가하면 검증기가 실패한다.
