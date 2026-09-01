# ShadowCrafter 개발 상태

마지막 갱신: 2026-09-01 (Asia/Seoul)

> 전체 진행률 추정: 약 55%. 이는 엔지니어링 작업량 기준이며 모델 성능 점수가
> 아닙니다. 모델은 아직 미출시 연구 후보이고 95% 목표는 아직 측정하지
> 않았습니다. 정확도는 private experimental release 업로드를 차단하지 않습니다.

## 완료된 기반 작업

- private GitHub `Odytssey/ShadowCrafter`와 9B private Hugging Face 저장소 생성
- 방어 우선 허용 정책, 데이터 거버넌스, 위협 모델, 릴리스 정책 수립
- 허가 증빙, exact allowlist, DNS pinning, 읽기 전용 메서드를 강제하는 수동
  HTTP/TLS 블랙박스 점검 프레임 구현
- 소스·설정·manifest·평가 증거는 로컬, 모델 weight는 승인 원격에만 두도록
  보관 정책과 동기화 스크립트 정렬
- 승인 artifact, exact host/path, DNS·peer pinning, 요청·응답 제한을 강제하는
  블랙박스 취약점 후보 탐지와 CLI 구현
- HSTS/CSP/CORS/cookie/cache/method/TLS 및 bounded body signal을 증거 기반으로
  판정하고 response body와 credential은 결과에 보존하지 않도록 검증

## 모델 경로

### ShadowCrafter-9B

- upstream revision: `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`
- H100에서 두 optimizer-step QLoRA 호환성 검증 완료
- preflight train loss `1.772` 확인(정확도 수치가 아님)
- LoRA-only adapter 파일 검증 완료
- 2,017건 기준선 전체 학습 job `shadowcrafter-9b-full-1474c34-v1` 실행 중
- 블랙박스 전용 레코드를 포함한 28,140건 확장 학습 job
  `shadowcrafter-9b-expanded-blackbox-264540a-v1`이 기준선 성공을 기다리는 중
- 고정 외부 평가와 릴리스 게이트는 미완료

## 데이터와 평가

- MITRE ATT&CK Enterprise/Mobile/ICS, CWE, CAPEC, OCSF, Splunk Security
  Content와 Odytssey 검토형 블랙박스 통제에서 출처가 고정된 방어형 학습
  레코드 28,140개 생성
- 기존 2,017건 대비 약 14.0배이며 threat intelligence 17,641건, detection
  engineering 4,060건, SIEM query 3,093건, secure code review 2,868건,
  CWE mapping 448건, black-box assessment 30건으로 구성
- exact/normalized 중복 0건, secret redaction 0건이며 CAPEC에서 CTI-Bench와
  겹친 3건만 hash-audited decontamination으로 제외
- 그래프 계보 누수를 피하기 위해 이 artifact는 명시적인 train-only로 표시하며
  내부 validation/test 점수에 사용하지 않음
- CTI-Bench 고정 revision의 5,610개 test row를 eval-only로 수집
- 무정답 및 안전·형식 실패를 제외한 5,533개 외부 평가 후보 생성
- CTI-Bench는 CC-BY-NC-SA-4.0이므로 학습 및 상업적 평가에 사용하지 않음
- 최종 28,140개 학습 레코드와 5,533개 CTI-Bench 평가 레코드의 exact/embedded
  오염 검사 결과 overlap 0건
- accuracy, balanced accuracy, macro-F1과 오염을 원시 예측에서 다시 계산하는
  고정 평가 프로토콜 구현 완료; 0.95 달성 여부는 보고하되 업로드는 차단하지 않음

## 다음 순서

1. 실행 중인 2,017건 기준선 학습을 완료하고 adapter 무결성을 검증
2. 고정 commit과 28,140건 train-only manifest로 9B 확장 학습 candidate 실행
3. 학습에 노출되지 않은 허용된 고정 외부 세트로 정확도·안전·슬라이스 평가
4. 실제 점수와 `95% target met` 상태를 적은 private Experimental Release 업로드
5. 실패 분석은 개발 세트에만 반영하고 blind test 오염 없이 후속 모델 개선

실제 95% 달성 여부는 사전에 보장할 수 없습니다. 목표 미달은 실제 점수와 함께
Experimental Release로 게시합니다. 다만 안전성, 라이선스, 개인정보, 계보 또는
artifact 무결성 통제에 실패하면 정확도와 관계없이 게시하지 않습니다.
