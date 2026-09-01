# 승인형 블랙박스 점검

ShadowCrafter의 블랙박스 하위 시스템은 소유자 승인을 받은 HTTP/TLS 자산의
**수동적 취약점 후보 탐지**를 수행합니다. 소스가 없어도 외부 응답에서 확인 가능한
전송·브라우저·쿠키·캐시·CORS·정보 노출 문제를 탐지하지만, 취약점 exploit,
payload 전송, 자격 증명 사용·검증, brute force, 서비스 거부, 상태 변경,
리디렉션 추적은 구현하지 않습니다. 결과는 확정 판정이 아니라 증거가 연결된
후보이며 항상 사람이 검토해야 합니다.

## 승인 문서

네트워크 요청 전에 자산 소유자가 승인한 immutable JSON 파일이 필요합니다. 런타임 `BlackBoxScope.authorization.evidence_sha256`은 이 파일의 정확한 바이트에 대한 SHA-256이어야 합니다. 파일의 승인 ID, 담당자, 기간, 대상, 메서드는 런타임 scope와 정확히 일치해야 합니다.

```json
{
  "schema_version": "1.0",
  "authorization_id": "AUTH-BB-001",
  "scope_id": "scope-bb-001",
  "approved_by": "asset-owner@example.test",
  "purpose": "Passive HTTP and TLS configuration review for an owned test service.",
  "allowed_targets": ["app.example.test"],
  "allowed_paths": ["/", "/health"],
  "safe_methods": ["HEAD", "GET", "OPTIONS"],
  "valid_from": "2026-09-01T00:00:00Z",
  "valid_until": "2026-09-01T04:00:00Z",
  "passive_read_only_only": true,
  "payloads_allowed": false,
  "redirects_allowed": false,
  "brute_force_allowed": false,
  "credential_testing_allowed": false,
  "exploit_execution_allowed": false,
  "denial_of_service_allowed": false,
  "state_changing_requests_allowed": false
}
```

승인 파일은 승인 시스템 또는 접근 통제된 vault에 보존합니다. Git, 모델 저장소, 프롬프트, 학습 데이터에 고객 승인 문서나 실제 대상 정보를 넣지 않습니다. 승인 만료 시 대기 중이던 요청도 실패하도록 각 요청 직전에 기간을 다시 확인합니다.

## 강제되는 네트워크 경계

- 대상은 `http` 또는 `https`의 표준 포트이며 사용자 정보, query, fragment가 없어야 합니다.
- 호스트는 exact hostname/IP 또는 명시적 CIDR allowlist로만 비교합니다. wildcard와 하위 도메인 자동 확장은 없습니다.
- 경로는 승인 JSON의 `allowed_paths`와 정확히 일치해야 합니다. 기본 경로는 `/` 하나입니다.
- hostname이 사설, loopback, link-local 또는 그 밖의 non-global 주소로 해석되면 해당 IP/CIDR도 승인 대상에 별도로 있어야 합니다. public/private 혼합 DNS 응답은 거부합니다.
- 검증한 IP에 직접 연결하면서 원래 hostname으로 TLS 인증서와 SNI를 검증합니다. 실제 peer IP가 고정한 주소와 다르면 DNS rebinding으로 간주해 중단합니다.
- 요청은 body, cookie, Authorization, 임의 사용자 header 없이 `GET`, `HEAD`, `OPTIONS` 중 승인된 메서드만 사용합니다.
- redirect를 따라가는 코드 경로가 없습니다. `Location`은 query와 fragment를 삭제한 관찰 증거로만 남습니다.
- scope 기준 최대 60 requests/minute와 concurrency 4를 넘을 수 없습니다. 런타임은 기본 timeout 5초, response body 64 KiB, header 16 KiB, 대상 8개, 요청 24개로 더 제한하며 hard maximum도 둡니다.

## 관찰 결과와 증거

다음과 같은 이미 수신된 메타데이터만 후보 판정에 사용합니다.

- TLS protocol, cipher, 인증서 fingerprint와 만료 시점
- HSTS, CSP, frame policy, content type 보호, referrer 및 cross-origin policy header
- HSTS의 malformed/zero/짧은 lifetime과 CSP report-only/unsafe script 정책
- 쿠키의 `Secure`, `HttpOnly`, `SameSite` 속성 및 cookie-setting 응답의 public cache
- CORS wildcard/null origin과 credential 조합
- `Server`, `X-Powered-By`, ASP.NET 제품·버전 노출
- 수동 `OPTIONS` 응답이 광고한 상태 변경 메서드와 `TRACE`/`CONNECT`
- 따라가지 않은 redirect가 승인 host scope 안인지 여부
- legacy TLS protocol, 약한 cipher와 30일 이내 인증서 만료
- 승인된 bounded `GET`에서 directory listing, runtime diagnostic page, verbose error의
  복수 고정 signature

응답 body는 결과에 포함하지 않습니다. 제한된 prefix의 바이트 수, SHA-256,
truncation 여부와 고정된 body signal ID만 기록합니다. `Set-Cookie` 값, CSP nonce/hash,
redirect query/fragment, 비허용 header는 evidence 생성 전에 삭제합니다. 각
`BlackBoxFinding`은 `EvidenceReference`를 통해 redacted evidence의 canonical
SHA-256과 연결됩니다. 제품 banner만으로 CVE를 단정하지 않으며, 정확한 제품·버전과
별도로 고정된 CVE 데이터베이스가 있을 때만 후보로 상관 분석합니다.

## 라이브러리 경계

broker는 `shadowcrafter.blackbox.assess_authorized_targets`(async) 또는 `run_authorized_assessment`(sync)에 검증된 `BlackBoxScope`, 승인 파일 바이트, exact URL 목록을 전달합니다. 기본 메서드는 `HEAD` 하나입니다. 이미 event loop가 있는 서비스는 async entry point만 사용해야 합니다.

CLI에서는 승인 파일과 별도의 runtime scope JSON을 전달하고, 결과를 새 private
파일로만 생성합니다. `--target`과 `--method`는 여러 번 지정할 수 있으며 기본
메서드는 `HEAD`입니다. body signature 탐지가 필요하면 승인 문서와 scope 양쪽에
`GET`이 포함되어 있어야 합니다.

```bash
shadowcrafter assess blackbox \
  --scope /secure/scope.json \
  --authorization /secure/authorization.json \
  --target https://app.example.test/ \
  --method GET \
  --method OPTIONS \
  --output reports/private/app-assessment.json
```

명령은 기존 output을 덮어쓰지 않으며 결과 파일 권한을 `0600`으로 설정합니다.

실제 배포에서는 호출자 인증, tenant별 scope 저장소, 승인 파일 vault, append-only 감사 로그, emergency stop을 이 라이브러리 바깥의 broker에서도 강제해야 합니다. 모델 출력으로 scope, 승인 파일, 대상 URL 또는 제한 값을 자동 생성하거나 확대해서는 안 됩니다.

## 검증

`tests/test_blackbox.py`는 실제 네트워크를 사용하지 않고 모의 resolver와 transport만
사용합니다. 승인 hash/기간, exact host/path, query 거부, private-address와 혼합 DNS,
peer 변경, redirect 차단, body/header/request 제한, redaction, concurrency와 rate
limit뿐 아니라 확장 header/TLS/body signature와 body 비보존도 회귀 검증합니다.
