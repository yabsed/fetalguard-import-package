# 실행 후 확인·보존·반출 심사

파일 확장자 대신 **내용과 생성 경로**로 구분한다. XGBoost 모델도 JSON이고, CSV에는 집계표와 개인별 행이 모두 있을 수 있다.

```text
결과/full-실행해시/
├── status.json
├── internal/                    # 현장 전용
│   ├── protocol.json · run_manifest.json
│   ├── data/ · splits/
│   ├── experiment_a/ · experiment_b/
│   ├── supplementary/ · official/
│   └── report/report.html · case_review.html · case_review.csv · case_*.png
└── export_review/               # 반출 심사 후보, 승인 전
    ├── report.html · report.md
    ├── *.csv · figures/
    ├── protocol_summary.json
    └── EXPORT_MANIFEST.json
```

| 구분 | 내부 전용 `internal/` | 심사 후보 `export_review/` |
|---|---|---|
| 보고서 | 전체 분석, 개별 설명 파일·내부 산출물 링크 | 집계 수치에서 새로 작성한 HTML·Markdown |
| 데이터 | 신호 캐시, 개인별 인자·EMR·라벨·분할·예측 | 충분한 집단의 표본 수·성능·CI |
| 설명 | 개인별 SHAP와 값, 정탐·오탐·미탐 파형 사례 | 평균 중요도, 집계 인자-판독 관계, 그룹 제거 차이 |
| 사이트 | 원 사이트 코드와 개별 자료 | S01 등 별칭과 선별된 집계 |
| 재현 정보 | 경로·파일명·원본별 해시·환경·전체 설정 | 허용한 설정·코드 패키지 해시·묶음 파일 해시 |
| 모델 | 가중치·학습 상태·체크포인트 | 모델 종류·선택 설정·비용 요약만 |

## 현장에서 할 일

1. `status.json`이 `complete`인지 확인하고 `internal/report/report.html`을 연다.
2. 인자 측정 가능성, 분석 누락, 학습 상한 도달, H1–H5의 비교 CI, 사이트별 오경보, 아웃컴 결측을 확인한다.
3. `internal/report/case_review.html`에서 Cat28의 TP/TN/FP/FN별 최대 2개 파형을 확인한다. 미리 정한 극단 점수 사례이며 대표 표본이 아니다. 개별 SHAP가 계산되지 않은 사례는 미산출로 표시한다. 개인별 설명·예측·모델은 내부에서 검토하고, ID를 지웠다는 이유만으로 이런 행을 반출 묶음에 추가하지 않는다.
4. `tools/validate_run.py`로 단계별 해시와 심사 묶음의 무결성을 검증한다.
5. `export_review/report.html`과 집계 CSV를 확인하고 이 묶음을 기관 심사에 제출한다. 승인된 파일만 시설 절차대로 반출한다.

`export_review`는 승인 여부를 뜻하지 않는다. `EXPORT_MANIFEST.json`의 상태는 `pending_institution_review`다. 자동 외부 전송 기능은 없다. 기관이 모델·개별 사례 등 추가 산출물의 반출을 승인한다면 별도 범위로 처리하며, 기본 묶음에는 포함하지 않는다.

## 자동 선별의 범위

심사 묶음은 고정된 파일명·필드 구조로 생성한다. 원래 report HTML, `run_manifest.json`, 임의 JSON, 원시 파일·개인 ID·원본 파일명·데이터별 해시·가중치를 복사하지 않는다. 사이트 별칭도 익명성을 보증하지 않으므로 심사 대상이다.

`export_min_mothers`의 기본값은 10이다. 산모 수와 필요한 양성/음성 산모 분모가 부족하거나 확인되지 않으면 해당 집계값을 억제한다. 전체 내부 분석은 그대로 남는다. 다른 표의 합계·구성·희귀 사건을 조합할 때의 노출 가능성까지 기관의 검토가 필요하며, 이 선별을 법적·기관 차원의 익명화 판정으로 간주하지 않는다.

SHAP 평균 중요도는 전체 테스트셋이 아니라 **실제 SHAP 계산 표본**의 산모 수로 선별한다. 아웃컴 관측/결측 집단의 나이·주수 등의 요약도 해당 변수가 실제 관측된 산모 수를 사용한다.

CV·seed별 보조표의 정상 기록 경보율·정상 관찰시간당 양성 창 수는 해당 행의 경보/무경보 산모 분모를 확인하지 않으므로 반출 묶음에서 공란으로 둔다. 이 지표는 별도로 분모를 심사한 주 비교표와 LOSO 표에서 확인한다. 현장 전용 표에는 원래 값이 남는다.

집계만으로 외부에서 임의의 새 하위군 분석·임계값 재선정·개인별 오류 조사를 할 수는 없다. 핵심 분석과 필요한 집계는 퇴실 전에 검토하고 내부 결과의 보존은 시설 정책에 따른다.

## 심사 묶음 주요 파일

- `model_comparison.csv`, `paired_comparisons.csv`: 동일 테스트 코호트 성능·차이·신뢰구간.
- `cv_summary.csv`, `seed_holdout_metrics.csv`, `seed_paired_cat28_minus_comparator.csv`: 반복·seed별 변동과 절제 방향.
- `feature_quality.csv`, `feature_response_train.csv`, `feature_importance.csv`: 측정 제약과 집계 설명.
- `cnn_capacity_validation.csv`, `cnn_capacity_summary.csv`, `cnn_selection.csv`, `normalization_matched.csv`: CNN 선택과 같은 width/seed 정규화 비교.
- `inference_cost.csv`: 모델별 추론 비용. 인자 추출·배치·IO 포함 범위와 함께 읽는다.
- `outcomes.csv`, `outcome_associations.csv`, `outcome_reading_cells.csv`, `outcome_missingness.csv`, `outcome_cohort_descriptors.csv`: 실제 전문의 판독-아웃컴 대응과 증분·결측 집단 특성.
- `emr_selection.csv`, `emr_matched_seeds.csv`, `loso.csv`, `official_reference.csv`: EMR 모델 선택·짝지은 비교, 사이트별 운영 성능, 별도 타깃·부분집합 참조.
- `hypothesis_evidence.csv`, `cohort_counts.csv`, `protocol_summary.json`: 질문별 근거와 분석 규모·설정 요약.
- `EXPORT_MANIFEST.json`: 실제 생성 파일 목록·SHA256·검토 대기 상태.

수행 불가 또는 선별 억제 항목은 빈 표/상태로 남을 수 있다. `report.html`은 심사 묶음 밖을 참조하지 않으므로 디렉토리 전체를 함께 다룬다. 원본 파일을 수동으로 추가하면 검증기가 실패하도록 되어 있다.
