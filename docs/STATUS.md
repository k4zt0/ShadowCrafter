# ShadowCrafter 개발 상태

마지막 갱신: 2026-09-01 (Asia/Seoul)

> 전체 진행률 추정: 약 40%. 이는 엔지니어링 작업량 기준이며 모델 성능 점수가
> 아닙니다. 모델은 아직 미출시 연구 후보이고 94% 목표는 아직 측정하지
> 않았습니다. 정확도는 private experimental release 업로드를 차단하지 않습니다.

## 완료된 기반 작업

- private GitHub `Odytssey/ShadowCrafter`와 9B private Hugging Face 저장소 생성
- 방어 우선 허용 정책, 데이터 거버넌스, 위협 모델, 릴리스 정책 수립
- 허가 증빙, exact allowlist, DNS pinning, 읽기 전용 메서드를 강제하는 수동
  HTTP/TLS 블랙박스 점검 프레임 구현
- 소스·설정·manifest·평가 증거는 로컬, 모델 weight는 승인 원격에만 두도록
  보관 정책과 동기화 스크립트 정렬

## 모델 경로

### ShadowCrafter-9B

- upstream revision: `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`
- H100에서 한 optimizer-step QLoRA 호환성 검증 완료
- train loss `3.349`, eval loss `3.565`
- LoRA-only adapter 파일 검증 완료
- 보안 학습 전체 run, 고정 외부 평가, 릴리스 게이트는 미완료

## 데이터와 평가

- MITRE ATT&CK Enterprise/Mobile/ICS에서 출처가 고정된 방어형 학습 레코드
  2,017개 생성
- 그래프 계보 누수를 피하기 위해 이 artifact는 명시적인 train-only로 표시하며
  내부 validation/test 점수에 사용하지 않음
- CTI-Bench 고정 revision의 5,610개 test row를 eval-only로 수집
- 무정답 및 안전·형식 실패를 제외한 5,533개 외부 평가 후보 생성
- CTI-Bench는 CC-BY-NC-SA-4.0이므로 학습 및 상업적 평가에 사용하지 않음
- accuracy, balanced accuracy, macro-F1과 오염을 원시 예측에서 다시 계산하는
  고정 평가 프로토콜 구현 완료; 0.94 달성 여부는 보고하되 업로드는 차단하지 않음

## 다음 순서

1. 통합 테스트와 독립 코드 리뷰를 완료하고 GitHub/Hugging Face 모델 카드를 갱신
2. 고정 commit과 train-only manifest로 9B 첫 보안 학습 candidate 실행
3. 학습에 노출되지 않은 허용된 고정 외부 세트로 품질·안전·슬라이스 평가
4. 실제 점수와 `94% target met` 상태를 적은 private Experimental Release 업로드
5. 실패 분석은 개발 세트에만 반영하고 blind test 오염 없이 후속 모델 개선

실제 94% 달성 여부는 사전에 보장할 수 없습니다. 목표 미달은 실제 점수와 함께
Experimental Release로 게시합니다. 다만 안전성, 라이선스, 개인정보, 계보 또는
artifact 무결성 통제에 실패하면 정확도와 관계없이 게시하지 않습니다.
