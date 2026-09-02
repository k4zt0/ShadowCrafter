# ShadowCrafter 개발 상태

마지막 갱신: 2026-09-02 (Asia/Seoul)

> v1.0 학습·평가·공개는 완료됐다. V2는 92,239건 대규모 corpus 재학습 단계다.
> 95%는 고정 외부 평가의 목표이며 보장 수치가 아니다.

## 완료된 기반 작업

- private GitHub `Odytssey/ShadowCrafter`와 9B public Hugging Face 저장소 구성
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
- v1.0 28,140건 QLoRA 학습과 LoRA-only adapter 검증 완료
- CTI-Bench 5,533건 accuracy `46.267847%`, balanced accuracy `11.099532%`,
  macro-F1 `6.748866%`
- public Hugging Face `KaztoRay/ShadowCrafter-9B`의 `v1.0` 게시 및 익명
  다운로드/SHA-256 검증 완료
- V2는 NIST Juliet 64,099건을 추가한 총 92,239건 corpus로 새 adapter를 학습한다.
- 공식 VS Code Colab 확장용 accountless notebook, 내장 source/data 검증,
  임시 checkpoint 완전성, CTI-Bench 동결 평가 및 측정 evidence 포함 로컬 export 구현 완료

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

## V2 다음 순서

1. NIST 공식 ZIP과 64,099개 testcase lineage 검증
2. 기존 corpus와 합친 92,239건의 중복·비밀·CTI-Bench 오염 검사
3. VS Code에서 Colab A100급 런타임을 연결하고 고정 commit으로 QLoRA V2 candidate 학습
4. 학습 완료 adapter 무결성 검증과 notebook 7번 셀의 CTI-Bench 재평가
5. 실제 점수와 `95% target met` 상태를 적은 public noncommercial v2.0 업로드
6. 모델·체크포인트·전체 원격 프로젝트를 `local_mirror/`에 증분 동기화

실제 95% 달성 여부는 사전에 보장할 수 없습니다. 목표 미달은 실제 점수와 함께
Official Release로 게시합니다. 다만 안전성, 라이선스, 개인정보, 계보 또는
artifact 무결성 통제에 실패하면 정확도와 관계없이 게시하지 않습니다.
V2는 v1.0 평가 문항이나 정답을 학습하지 않고 별도 immutable candidate로 진행한다.
