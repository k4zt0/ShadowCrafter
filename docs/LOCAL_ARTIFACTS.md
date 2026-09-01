# 로컬 Artifact 및 복구 정책

## 1. 목표

ShadowCrafter의 소스, 데이터, manifest와 릴리스 증거는 로컬 PC에서 감사할 수
있어야 합니다. 사용자의 최신 지침에 따라 기반 모델·체크포인트·최종 weight
파일은 로컬에 보관하지 않으며, 승인된 원격 저장소에서 복구 가능해야 합니다.

현재 로컬 보관 범위는 다음과 같습니다.

- 소스, 설정, 문서, lockfile과 manifest의 로컬 Git 이력
- 허가된 각 데이터 snapshot의 실제 파일 또는 법적 제한이 있는 경우 접근 통제된 로컬 사본과 제한 기록
- 기반 모델 revision, tokenizer revision, 원격 checkpoint와 릴리스의 전체 hash manifest
- 평가 입력의 허용된 사본, 평가 결과, 안전성 증거, 로그와 보고서
- 모든 파일을 연결하는 hash, provenance, 라이선스와 원격 commit 기록

가중치가 원격에만 있으므로, 완료 표시는 두 개의 독립적인 승인 원격 사본과
fresh-cache 복원 검증이 있을 때만 허용합니다. 로컬 manifest만으로 모델을 복구할
수 있다고 표현하지 않습니다.

## 2. 권장 디렉터리

```text
ShadowCrafter/
├── artifacts/
│   ├── evaluations/<model>/<run_id>/
│   ├── manifests/<model>/<run_id>/
│   ├── preflight/
│   └── environment/
├── data/
│   ├── staging/
│   ├── raw/<dataset>/<snapshot>/
│   ├── interim/<dataset>/<snapshot>/
│   ├── processed/<dataset>/<snapshot>/
│   ├── snapshots/<dataset>/<snapshot>/
│   └── quarantine/
├── manifests/
│   ├── data/
│   ├── runs/
│   └── releases/
├── reports/
│   ├── generated/
│   └── private/
└── src/, configs/, scripts/, tests/, docs/ ...
```

대용량·민감 디렉터리는 `.gitignore` 대상입니다. `manifests/`의 비민감 소형 메타데이터만 Git으로 추적할 수 있으며, manifest에 로컬 절대 경로·토큰·개인정보·악성 payload를 넣지 않습니다.

## 3. 자산별 원본

| 자산 | 권위 있는 로컬 위치 | 원격 역할 |
|---|---|---|
| 소스/설정/문서 | 작업 트리 + 로컬 Git object database | private GitHub 협업 복제 |
| 데이터 | `data/snapshots/` + data manifest | 허용되는 경우에만 private 복제 |
| 기반 모델 | 로컬에는 revision/hash manifest만 | 고정 upstream revision + 원격 cache |
| run checkpoint | 로컬에는 run/file manifest만 | 원격 GPU + 승인된 독립 원격 사본 |
| 평가 | `artifacts/evaluations/` + evaluation card | 필요한 비민감 결과만 공유 |
| 완료 모델 | 로컬에는 release/evaluation manifest만 | 9B private Hugging Face + 독립 원격 복구 사본 |

## 4. 파일 명명과 불변성

- 모델, dataset, run, release ID에는 경로 안전한 문자만 사용합니다.
- run ID는 모델 계열, UTC 시작 시각과 고유 ID를 연결합니다.
- snapshot과 release 디렉터리는 승격 후 덮어쓰지 않습니다. 변경은 새 version을 만듭니다.
- `latest` 같은 가변 이름은 사람이 읽는 포인터로만 쓰고 manifest와 자동화는 immutable ID를 참조합니다.
- 모든 시각은 UTC ISO-8601로 기록합니다.

## 5. Manifest 최소 요건

release manifest에는 다음을 포함합니다.

- model family/version, base model ID/revision와 tokenizer revision
- Git commit, dirty-tree 상태, 환경 lock/container digest
- data snapshot ID와 manifest hash, split/evaluation ID
- training run ID, seed, 핵심 hyperparameter와 완료 상태
- 상대 경로, 바이트 크기, SHA-256을 가진 전체 파일 목록
- 평가·안전성·라이선스 검토 상태와 승인자
- GitHub/Hugging Face private 저장소와 원격 commit SHA(게시 후)
- 생성 시각, schema version, 이전/rollback release

manifest 자체도 hash하고 가능하면 별도의 서명 또는 신뢰 가능한 투명성 기록으로 보호합니다. 임의 pickle처럼 로딩 시 코드를 실행할 수 있는 형식은 피하고 안전한 직렬화 형식을 우선합니다.

## 6. 원격 학습 생명주기

### 전송 전

1. Git 작업 상태와 commit을 기록하고 비밀 검사를 수행합니다.
2. 기반 모델, 데이터 snapshot, 코드, 환경 image를 immutable revision/digest로 고정합니다.
3. 필요한 파일만 upload manifest에 열거합니다. 데이터/소스는 전송 전 로컬 SHA-256을,
   로컬에 둘 수 없는 model/weight는 원격에서 계산한 SHA-256을 로컬 manifest pin과
   독립 원격 검증으로 확인합니다.
4. 원격 계정·호스트 키·목적 경로·가용 용량·암호화·보존 정책을 확인합니다.
5. 개인 SSH 키나 장기 토큰을 작업 번들에 포함하지 않습니다.

### 학습 중

- 원격 작업은 전용 경로와 최소 권한 계정에서 수행합니다.
- 주기 checkpoint마다 completeness marker와 hash를 만들되, 쓰는 중인 파일은 회수하지 않습니다.
- 로그에서 토큰, 로컬 경로의 사용자 정보, 원문 비밀과 과도한 데이터 샘플을 마스킹합니다.
- 원격 GPU의 weight를 유일한 사본으로 두지 않으며, 독립 원격 복제 전 삭제하지 않습니다.

### 검증 증거 회수와 원격 승격

1. 완료 marker가 있는 run의 전체 파일 inventory와 SHA-256을 원격에서 생성합니다.
2. manifest·로그·평가 증거만 로컬 staging으로 회수하고 hash를 비교합니다.
3. shard index, tokenizer/config, 안전한 로딩과 추론 smoke test를 원격에서 검증합니다.
4. 불완전하거나 불일치한 파일은 원격 격리하고 같은 경로에서 고치지 않습니다.
5. 검증된 candidate만 immutable 원격 release 경로로 원자적으로 승격합니다.
6. 모델 weight를 private Hugging Face 또는 승인된 독립 원격 저장소에 복제합니다.
7. fresh-cache 복원과 전체 hash 검증 뒤에만 원격 임시 checkpoint 정리를 승인합니다.

## 7. GitHub와 Hugging Face 동기화

### private GitHub

소스, 설정, 테스트, 문서와 비민감 manifest만 `Odytssey/ShadowCrafter`에 커밋합니다. 가중치, 데이터, 악성 샘플, 생성 보고서, 키와 토큰은 커밋하지 않습니다. push 전 저장소 소유자와 `Private` 가시성을 확인하고, branch protection과 최소 권한을 적용합니다.

### private Hugging Face

게이트를 통과한 모델은 계열별 private 저장소에만 업로드합니다. 업로드 전에 원격
release의 전체 inventory를 확인하고, 로컬 인증을 원격에 복사하지 않는 memory-only
publisher로 하나의 parent-pinned commit을 생성합니다. 게시 직후에는 로컬 filesystem
cache 없이 Hub byte stream을 검증하고, 이어서 fresh remote cache에서 받은 파일의
hash를 로컬 manifest와 비교합니다. 데이터셋은 별도의
권리·개인정보·재배포 승인을 통과한 경우에만 분리된 private 저장소를 사용합니다.

원격 저장소의 파일을 웹 UI에서 직접 수정하지 않습니다. 불가피한 model card 수정도 로컬에 먼저 반영하고 양쪽 commit을 lineage에 기록합니다.

## 8. 용량과 백업

- 학습 전에 기반 모델, optimizer state, checkpoint 수, 데이터 snapshot, 임시 복사본과 안전 여유를 포함해 필요한 용량을 계산합니다.
- 사용 가능한 공간이 보수적 임계값 아래면 새 run이나 회수를 시작하지 않습니다.
- 백업은 암호화하고 복구 키를 데이터와 분리하며 정기적으로 restore test를 수행합니다.
- cache는 재생성 가능하지만 삭제 전에 manifest가 가리키는 파일이 아닌지 확인합니다.
- checkpoint 정리는 release와 직접 연결되지 않고 보존 기간이 지난 파일만 검토·승인 후 수행합니다.

## 9. 복구 점검표

fresh approved remote cache를 사용한 격리 검증에서 다음이 가능해야 완료 상태입니다.

- 고정된 Git commit에서 코드·설정·문서를 checkout
- 로컬 데이터와 기반 모델 manifest의 모든 hash 검증
- 원격 release의 tokenizer/config/weight shard를 로딩
- 최소 추론 및 대표 평가의 재실행
- 결과를 원래 evaluation card와 비교
- 어느 데이터·코드·기반 모델·run이 release를 만들었는지 역추적

복구 테스트 결과와 마지막 성공 시각을 release manifest에 기록합니다.

## 10. 비밀과 민감 자료

로컬 저장은 무제한 접근을 뜻하지 않습니다. SSH 키와 서비스 토큰은 프로젝트 디렉터리 밖의 운영체제 키 저장소 또는 승인된 secret manager에 두고 권한을 최소화합니다. 악성 샘플과 기밀 데이터는 별도 암호화 볼륨, 접근 로그, 검역 정책을 사용합니다. 민감 보고서와 원천 로그는 모델 가중치·일반 백업과 분리하고 보존 종료 시 안전하게 폐기합니다.
