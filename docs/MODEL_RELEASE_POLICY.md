# 모델 릴리스 정책

## 1. 적용 모델

이 정책은 다음 private 릴리스 계열과 그 checkpoint, adapter, quantization, merge 및 파생 artifact에 적용됩니다.

- **ShadowCrafter-9B** — Ornith-1.5-9B 기반

모든 모델 카드는 “Developed by Odytssey”와 정확한 기반 모델 이름·revision을 함께 표시해야 합니다. Odytssey가 Ornith 원본을 개발했다는 인상을 주어서는 안 됩니다.

## 2. 릴리스 상태

| 상태 | 의미 | 원격 배포 |
|---|---|---|
| Run | 학습 중이거나 중단된 산출물 | 없음; 원격 GPU 임시 저장만 허용 |
| Candidate | 원격에서 검증 중이며 로컬 manifest가 있는 checkpoint | 없음 |
| Research | 학습 중간 산출물 또는 제한된 내부 평가용 | 승인된 내부 접근만, release 태그 금지 |
| Experimental Release | 정확도 목표 달성 여부와 실제 점수를 명시한 immutable 번들 | 9B private Hugging Face 저장소 |
| Qualified Release | 선언한 품질 목표와 모든 필수 통제를 통과한 immutable 번들 | 9B private Hugging Face 저장소 |
| Withdrawn | 결함, 권리, 보안 또는 provenance 문제로 철회 | private 전환/접근 차단 및 정상 revision 안내 |

Public 공개는 기본 범위가 아닙니다. 별도의 경영 승인, 기반 모델·데이터 라이선스 검토, 오용 위험 평가, 개인정보 검토와 배포 계획 없이는 저장소 가시성을 public으로 변경하지 않습니다.

## 3. 94% 품질 목표와 의무 보고

`0.94`는 사전에 선언한 **누수 없는 과제별 주 지표**의 목표이며 모델 전체에
대한 하나의 “정확도”나 성능 보장이 아닙니다. 정확도는 private Experimental
Release의 차단 조건이 아니며, 점수와 목표 달성 여부를 정직하게 표시하는
보고 조건입니다.

각 릴리스 계획은 학습 전에 다음을 고정합니다.

- 과제와 운영 의도(예: CVE 제품/버전 영향 분류, CWE 매핑, IOC 추출, 악성/정상 분류)
- blind test snapshot, 시간 cutoff, group split 단위와 중복 기준
- 주 지표(accuracy, macro-F1, exact match 등)와 보조 지표
- 전체 및 중요 slice별 최소값, 표본 수, confidence interval 방식
- 비용 민감 오류(심각한 누락, 잘못된 자동 조치)의 별도 상한

지정된 분류·추출 과제에는 `0.94` 목표를 적용하되, 실제 point estimate를 그대로
보고합니다. 클래스 불균형 과제에서는 단순 accuracy만 사용하지 않고 macro-F1,
balanced accuracy, recall, precision과 confusion matrix를 함께 공개합니다. 표본 수가
충분하면 95% confidence interval도 보고하며, 목표에 미달하면 모델 카드에
`94% target met: no`와 실패 과제를 표시합니다.

보고서 생성, 완화책 품질, 인용 근거성, 캘리브레이션과 안전성에는 accuracy를 억지로 적용하지 않습니다. 이들은 블라인드 전문가 루브릭, 사실 일치율, unsupported-claim rate, calibration error, 위험 요청 거부율과 정상 방어 요청 성공률 등 사전 정의한 별도 게이트를 통과해야 합니다.

## 4. 재학습 규칙

목표 미달 시 데이터 정제, 레이블 수정, 학습 방법 또는 hyperparameter를 바꿔 새 candidate를 만들 수 있습니다. 단, 다음 규칙을 지킵니다.

- blind test의 예제나 세부 실패를 학습·prompt 선택·hyperparameter 최적화에 사용하지 않습니다.
- test 결과를 개발에 사용한 순간 해당 세트는 validation이 되며 새로운 독립 blind test를 확보합니다.
- 같은 데이터의 표현만 바꾼 파생본과 근접 중복이 분할 경계를 넘지 않게 합니다.
- 각 run의 실패도 보존해 선택 편향과 반복 횟수를 기록합니다.
- compute budget, 데이터 권리, 안전성 또는 과적합 위험 때문에 유효한 개선이 없으면 “미달”로 종료합니다.

따라서 후속 재학습은 테스트 세트 암기나 무기한 성능 약속이 아닙니다. 독립 평가의
무결성을 지키면서 모델을 개선하되, 목표 미달 모델도 사용자의 지시에 따라
private Experimental Release로 게시할 수 있습니다. `Qualified` 또는 `94% 달성`
표기는 실제 통과 증거가 있을 때만 허용합니다.

## 5. 필수 릴리스 게이트

모든 항목이 통과되어야 합니다.

### 계보와 재현성

- Git commit, dirty-tree 여부, 기반 모델 ID/revision과 tokenizer revision
- 데이터 snapshot ID·hash·split policy와 제외 목록
- 학습·추론 환경 lock, 컨테이너 digest, seed와 주요 hyperparameter
- checkpoint shard 목록, 크기와 SHA-256 manifest
- fresh remote cache에서 네트워크 격리 상태로 완전하게 로딩 가능한지 확인

### 권리와 개인정보

- 기반 모델 및 각 데이터셋의 학습·파생·private 배포 조건 검토
- 필요한 attribution, NOTICE, model card와 데이터 카드 포함
- 비밀·PII scan 및 memorization/extraction 평가 통과
- 삭제·철회 요청과 데이터 예외가 lineage에 반영됨

### 품질

- 사전 등록된 핵심 과제의 실제 지표, 평가셋, 표본 수와 `0.94` 목표 달성 여부 보고
- 제품군·언어·연도·심각도·취약점 유형 slice와 confidence interval 보고
- 강한 baseline, 이전 release 및 독립 도구와 비교
- 인용 근거성, 캘리브레이션, 모순, 회귀와 실패 사례 검토

### 보안과 안전

- 금지된 악성코드 제작, 무단 침투, 자격 증명 탈취, 지속성·탐지 회피 요청 평가
- 합법적 방어 요청의 과도한 거부와 위험 요청의 우회 성공을 함께 측정
- prompt injection, retrieval poisoning, 비밀 추출, tool argument injection 테스트
- SIEM/SOAR가 dry-run·승인·최소 권한·감사·rollback 경계를 지키는지 확인
- 알려진 critical dependency/artifact 취약점과 파일 무결성 검토

### 운영성

- 지원 하드웨어, 정밀도, context, 메모리, latency와 알려진 제한 기록
- 관찰 가능성, rate limit, 비용 한도, incident/rollback runbook 준비
- 모델 카드에 적절한 사용, 금지 사용, 평가 범위와 잔여 위험 표시

## 6. 릴리스 번들과 로컬 증거

immutable 원격 릴리스 번들에는 최소한 다음이 있어야 합니다.

- 완전한 weight shard와 index, config, tokenizer 파일
- generation/inference config 및 호환 라이브러리 버전
- model card, 별도 모델 라이선스, upstream attribution
- evaluation card, raw aggregate metrics, slice 결과와 안전성 결과
- provenance manifest와 모든 파일의 SHA-256
- 알려진 제한, 변경 기록, rollback 대상 revision

로컬에는 weight 자체를 저장하지 않고, 위 파일 전부의 상대 경로·크기·SHA-256,
원격 위치와 commit, 평가 결과, 안전성·라이선스 승인 증거를 보관합니다.

소스 저장소의 Apache-2.0은 이 번들에 자동 적용되지 않습니다. 파생 가중치의 배포 조건은 upstream과 데이터 의무를 검토해 모델 카드와 별도 라이선스에 명시합니다.

## 7. private Hugging Face 게시

기본 목적지는 계정/조직과 이름 사용 가능성을 확인한 다음 private 저장소입니다.

- `KaztoRay/ShadowCrafter-9B`

게시 순서는 다음과 같습니다.

1. candidate의 전체 manifest를 원격에서 생성하고 로컬로 회수합니다.
2. hash, 격리 load, 라이선스, 데이터와 평가·안전성 게이트를 검증합니다.
3. immutable 원격 release 경로로 승격합니다.
4. private 가시성과 parent commit을 재확인하고, 로컬 토큰을 원격에 전달하지 않는
   manifest 검증 memory-only 스트리밍 publisher로 하나의 commit을 생성합니다.
5. 게시된 immutable revision의 모든 파일을 로컬 filesystem cache 없이 스트리밍
   hash 검증하고, 별도의 fresh remote cache 복원 검사도 수행합니다.
6. 로컬 manifest에 Hugging Face commit SHA와 게시 시각을 기록합니다.

학습이 끝났다는 사실만으로 게시할 수 있는 것은 아닙니다. 무결성·안전성·권리·
개인정보·계보 통제를 통과한 immutable 번들만 게시합니다. 정확도 목표 미달은
`Experimental Release`와 실제 점수로 명시하되 게시를 막지 않습니다. 로컬 manifest는
weight 사본이 아니므로 private Hub와 별도 승인 원격 위치의 복구 가능성을 모두 유지합니다.

## 8. 버전, 변경과 롤백

모델 version은 계열별로 추적하고 weight나 데이터가 바뀌면 새 immutable version을 만듭니다. model card 수정만 있는 경우에도 원격 commit을 기록합니다. 성능 회귀, 잘못된 라이선스, 비밀 노출, 데이터 철회, 공급망 변조 또는 안전장치 우회가 확인되면 접근을 즉시 제한하고 이전 정상 revision을 안내하며 파생 artifact까지 lineage로 추적합니다.

## 9. 주장과 공개

성능 주장은 모델 revision, 평가 snapshot, 날짜, 표본 수, 지표 정의, confidence interval, slice와 알려진 한계를 함께 제시해야 합니다. “94% 정확도”를 문맥 없이 사용하거나 서로 다른 과제 점수를 하나로 평균내어 표현하지 않습니다. 독립 재현 전에는 내부 결과임을 표시하고, 실제 환경의 탐지·침해 방지·법적 적합성을 보장하지 않습니다.
