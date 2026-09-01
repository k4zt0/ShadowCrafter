# ShadowCrafter

Odytssey가 개발하는 방어 우선(defensive-first) 사이버보안 특화 LLM 프로젝트입니다. ShadowCrafter는 Ornith-1.5-9B를 기반으로, 허가된 환경에서 취약점·CVE 분석, 보안 리포트 작성, 악성코드 징후 탐지, 보안 지식베이스 구축, SIEM/SOAR 분석 보조를 수행하도록 연구·학습합니다.

> **상태:** 초기 비공개 개발 단계입니다. 현재 문서에 적힌 기능과 목표 수치는 출시 또는 성능 보장이 아닙니다.

검증된 최신 진행 내역은 [docs/STATUS.md](docs/STATUS.md)에 기록합니다.

## 안전 경계

ShadowCrafter는 다음 원칙을 따릅니다.

- 소유자 또는 운영자의 명시적 허가를 받은 시스템과 격리된 샌드박스에서만 공격 기법을 시험합니다.
- 실환경 무단 침투, 자격 증명 탈취, 지속성 확보, 피싱, 파괴 행위, 악성코드 제작·배포, 탐지 회피를 지원하지 않습니다.
- 악성코드는 정적·동적 분석, 분류, IOC 추출, 대응 권고 등 방어 목적으로만 다루며 격리된 저장소와 실행 환경을 사용합니다.
- SIEM/SOAR 결과는 제안 또는 승인 대기 작업으로 취급합니다. 차단, 격리, 삭제처럼 영향이 큰 조치는 사람이 검토하고 승인해야 합니다.
- 모델 출력은 사실 검증과 전문가 검토가 필요한 분석 보조 자료이며 보안 판정의 단독 근거가 아닙니다.

자세한 허용 범위는 [ACCEPTABLE_USE.md](ACCEPTABLE_USE.md), 보안 가정은 [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md)를 참고하십시오. 승인형 수동 HTTP/TLS 점검의 강제 경계와 승인 문서 형식은 [docs/BLACK_BOX_ASSESSMENT.md](docs/BLACK_BOX_ASSESSMENT.md)에 있습니다.

## 모델 계열

| ShadowCrafter 계열 | 기반 모델 | 주 용도 | 배포 기본값 |
|---|---|---|---|
| ShadowCrafter-9B | Ornith-1.5-9B | 저지연 분류, 추출, 로컬 분석 보조 | 비공개 |

모델 카드에는 반드시 “Developed by Odytssey”와 기반 모델 계보를 함께 표시합니다. 이는 Odytssey가 Ornith 원본 모델을 만들었다는 의미가 아니라, 해당 기반 모델에서 ShadowCrafter를 파인튜닝했다는 의미입니다. 기반 모델, 토크나이저, 데이터셋 및 파생 가중치의 사용·배포에는 각각의 별도 라이선스가 적용됩니다.

## 목표 기능

- 소스·설정·컨테이너·클라우드 구성을 실행하지 않는 화이트박스 취약점 후보 탐지
- 승인 증빙, exact allowlist, DNS pinning, 읽기 전용 HTTP 메서드와 속도 제한을
  강제하는 블랙박스 취약점 후보 탐지
- CVE/CWE/CVSS와 영향을 받는 자산의 연결, 우선순위화 및 완화책 제안
- 침해 징후, 로그, 경보, 파일 메타데이터의 방어적 분석과 IOC 추출
- 재현 가능한 보안 리포트와 검증 가능한 출처·불확실성 기록
- 승인된 보안 문서를 기반으로 한 버전 관리형 보안 지식베이스
- SIEM 경보 보강 및 SOAR 플레이북 초안 생성(항상 사람의 승인 필요)
- 명시적으로 승인된 격리형 실습 환경에서의 레드팀·블루팀 평가

## 정확도 목표와 릴리스 표기

“94% 이상”은 범용 해킹 능력이나 모든 보안 업무에 대한 단일 정확도 약속이
아닙니다. 사전에 고정한 누수 없는 테스트 세트에서 지정 과제의 목표 달성 여부를
표시하는 수치입니다. 실제 점수는 높거나 낮은 그대로 보고하며, `0.94` 미달만으로
private experimental release 업로드를 막지 않습니다. 생성형 보고서, 안전성,
캘리브레이션처럼 accuracy가 부적절한 과제는 별도의 루브릭을 사용합니다.

테스트 세트를 보고 재학습하거나 임계값에 도달할 때까지 같은 테스트 세트에
과적합하지 않습니다. 중복·근접 중복 제거, 출처/프로젝트/시간 분리, 블라인드
평가, 신뢰구간과 실패 사례 보고가 필수입니다. 목표 미달 릴리스에는
`Experimental`, 실제 점수, 평가 범위와 `94% target met: no`를 명시합니다. 정확도와
별개로 무결성·라이선스·개인정보·안전성에 실패한 artifact는 업로드하지 않습니다.
자세한 기준은 [docs/MODEL_RELEASE_POLICY.md](docs/MODEL_RELEASE_POLICY.md)에 있습니다.

## 로컬·원격 자산 보관

사용자의 최신 보관 지침에 따라 로컬 PC에는 기반 모델과 체크포인트 가중치를
저장하지 않습니다. 로컬은 소스·설정·데이터·평가 증거·manifest의 원본이며,
모델 가중치는 승인된 원격 학습 서버와 private Hugging Face에 보관합니다.

| 자산 | 로컬 보관 | 비공개 원격 보관 |
|---|---|---|
| 소스, 설정, 문서, 소형 manifest | 이 Git 작업 트리와 로컬 Git 기록 | GitHub `Odytssey/ShadowCrafter` |
| 기반 모델, 체크포인트, 최종 가중치 | 저장하지 않음; hash와 계보 manifest만 보관 | 원격 GPU와 9B private Hugging Face 저장소 |
| 원천·가공 데이터와 악성 샘플 | `data/` 아래의 접근 통제·격리 저장소 | 라이선스와 보안 검토가 허용할 때만 private 저장소 |
| 평가 결과와 릴리스 증거 | 로컬 평가 번들과 checksum manifest | 필요한 비민감 자료만 9B private 모델 저장소 |

Hugging Face 비공개 저장소는 `KaztoRay/ShadowCrafter-9B`만 사용합니다. 완성된 모델은 로컬로 복사하지 않고 원격에서 hash·안전 로딩·평가를 검증한 뒤 private Hub에 복제합니다. 로컬 manifest에는 원격 commit과 전체 checksum을 기록합니다. 가중치를 원격에서 지우기 전에는 서로 독립적인 원격 복구 사본과 복원 검증이 필요합니다.

대용량 가중치·데이터·악성 샘플·비밀은 Git에 커밋하지 않습니다. 저장 구조, 동기화 순서, 복구 기준은 [docs/LOCAL_ARTIFACTS.md](docs/LOCAL_ARTIFACTS.md)를 따릅니다.

## 데이터와 학습 원칙

많은 자료를 수집하는 것보다 사용 권리, 출처, 품질, 최신성, 개인정보 최소화, 중복 제거와 평가 오염 방지가 우선입니다. Hugging Face 및 보안 자료는 허용 목록과 데이터 카드 검토를 통과한 스냅샷만 학습에 사용합니다. 비밀, 탈취 자격 증명, 불법 취득 자료, 배포가 제한된 데이터는 제외합니다. 세부 정책은 [docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md)를 참고하십시오.

현재 확장 train-only snapshot은 8개 승인 소스에서 28,140건을 포함하며, 고정된
CTI-Bench 5,533건과의 exact/embedded 오염 검사를 통과했습니다. 소스별 수량,
해시, 제외 근거와 아직 학습하지 않은 후보는
[docs/DATASET_EXPANSION.md](docs/DATASET_EXPANSION.md)에 기록합니다.

원격 GPU 학습은 재현 가능한 고정 manifest로 실행하며, 로그·평가 번들·checksum
manifest만 로컬로 회수합니다. 체크포인트 가중치는 로컬로 내려받지 않습니다.
SSH 키, Hugging Face 토큰, GitHub 토큰은 저장소나 학습 이미지에 포함하지 않습니다.

## 저장소와 릴리스 흐름

1. 로컬에서 변경하고 비밀·대용량 산출물이 없는지 검사합니다.
2. 비공개 GitHub 저장소에 작은 단위로 검토 가능한 커밋을 푸시합니다.
3. 고정된 데이터·코드·환경 manifest로 승인된 원격 GPU에서 학습합니다.
4. 원격 가중치를 제자리에서 검증하고 로컬에는 manifest·로그·평가 증거만 회수합니다.
5. 통과한 모델만 원격에서 9B private Hugging Face 저장소에 업로드하고 다시 검증합니다.
6. 로컬 manifest에 Git commit, 데이터 snapshot, 기반 모델 revision, 평가 결과, 원격 commit을 연결합니다.

어떤 원격 서비스에도 업로드하기 전에 해당 계정, 조직, 저장소 가시성이 정확한지 별도로 확인해야 합니다.

## 보안 및 라이선스

취약점이나 모델 안전 문제는 공개 이슈에 세부 내용을 남기지 말고 [SECURITY.md](SECURITY.md)의 비공개 신고 절차를 사용하십시오. 소스와 원본 문서는 별도 표시가 없는 한 Apache-2.0입니다. Ornith 가중치, 토크나이저, 데이터셋, 악성 샘플, 피드 및 파생 모델은 이 저장소의 Apache-2.0 허가 대상이 아니며 각 아티팩트의 조건을 따릅니다.

---

## English summary

ShadowCrafter is a defensive-first cybersecurity LLM project developed by Odytssey and fine-tuned from Ornith-1.5-9B. Its intended uses include authorized vulnerability and CVE analysis, defensive malware triage, evidence-grounded reporting, security knowledge bases, and human-approved SIEM/SOAR assistance. Offensive evaluation is limited to explicitly authorized, isolated sandboxes; unauthorized access, malware development or deployment, credential theft, persistence, destructive actions, and evasion are prohibited.

The local workstation is the source of truth for code, approved data, manifests, and evaluation evidence. At the operator's direction it does not retain base-model or checkpoint weights; those remain on approved remote storage and private per-model Hugging Face repositories with locally retained hashes and provenance. The “94%” value is a reported target for predeclared, leakage-free, task-specific metrics—not a universal accuracy claim, performance guarantee, or private-upload threshold. Below-target models may be published only as clearly labeled private Experimental Releases; integrity, safety, licensing, privacy, and provenance failures still block publication.
