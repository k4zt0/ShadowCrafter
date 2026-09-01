# ShadowCrafter-9B 확장 데이터셋

## 고정 학습 snapshot

- dataset ID: `security-expanded-20260901-v7-train-only`
- 생성 시각: `2026-09-01T11:51:35.154458+00:00`
- 학습 레코드: 28,110
- validation/test/evaluation 레코드: 0/0/0
- train SHA-256: `63d2e2f3bbe7efa0a6a1487977a1dc1f30055e2dc71ce1e3df986d6ad04a1179`
- prepared manifest SHA-256: `9b1fc8f14abce9505cddb75d84c58530cf39218c971c7c05ecd9914d71423b89`
- dataset fingerprint: `a1a46d40b692b6478491f8d14b045bb79ed7c8cf652aae202a14e7f4821ea8b1`
- source registry canonical SHA-256:
  `bdc69660611ee64db3aa5d3279e9859c81a64c08ba55bd46d0c51fe25fd09405`
- source registry raw-file SHA-256:
  `a1dd3af07b29bc72cc8145f6d1f608384c120f6eea2ebc653a9d24c0f04fb2c7`

이 snapshot은 group lineage를 유지한 train-only 자료다. 내부 validation/test 점수를
제공하지 않으며 정확도는 별도의 eval-only CTI-Bench snapshot으로만 측정한다.

## 승인 소스와 수량

| 소스 | 레코드 | 고정 revision | 용도 |
|---|---:|---|---|
| MITRE ATT&CK Enterprise | 19,079 | HTTP ETag 고정 | procedure mapping, detection, mitigation |
| MITRE ATT&CK Mobile | 1,854 | HTTP ETag 고정 | procedure mapping, detection, mitigation |
| MITRE ATT&CK ICS | 768 | HTTP ETag 고정 | procedure mapping, detection, mitigation |
| MITRE CWE | 1,699 | `Thu, 30 Apr 2026 09:15:04 GMT` | secure-code mitigation |
| MITRE CAPEC | 1,617 | `Tue, 24 Jan 2023 18:32:31 GMT` | mitigation, CAPEC→CWE mapping |
| Splunk Security Content | 1,987 | `ad15a0a3cb3ff29dca19160dd5bce30ebad89f78` | production SPL detections |
| OCSF schema | 1,106 | `40a1511e014da94d2d7a2ff964089425d0d479dd` | SIEM event normalization schema |

Splunk에서는 `production` detection만 읽고 experimental/deprecated rule, test payload,
attack simulation과 별도 attack-data를 제외했다. OCSF에서는 dictionary, category,
event/object/profile/extension schema만 읽고 예제 로그와 실행 코드를 제외했다.

## 과제 분포

| 과제 | 레코드 |
|---|---:|
| threat intelligence | 17,641 |
| detection engineering | 4,060 |
| SIEM query/schema | 3,093 |
| secure code review | 2,868 |
| CWE mapping | 448 |

exact duplicate, normalized duplicate와 secret redaction은 각각 0건이다. 데이터 양을
인위적으로 맞추기 위한 oversampling은 적용하지 않았다.

## 평가 오염 통제

CTI-Bench revision `9237e1636ee3e168fbe5ebdcc1c571de0525e568`의 5,533개
eval-only case와 normalized exact/containment 검사를 수행했다. CAPEC 완화책 3건이
겹쳐 레코드 ID와 학습 콘텐츠 hash만 남기고 제외했다. benchmark 문장은
decontamination manifest에 저장하지 않았다.

- 제외 전 CAPEC: 1,620
- 제외 후 CAPEC: 1,617
- 최종 검사: train 28,110 / evaluation 5,533 / overlap 0
- CTI-Bench eval JSONL SHA-256:
  `e78c90f1d4d9f75cd9a8011cad80b9614654f1d4a86c536144da0bc79e85a14f`

## 조사했지만 이번 SFT에서 제외한 자료

- Trendyol Cybersecurity Instruction Tuning Dataset: 53,202건과 Apache-2.0 표기는
  확인했지만, 데이터 카드가 언급한 500K+ 원출처의 record-level 권리·계보가 없어
  검역 상태를 유지한다.
- SecureCode: CC-BY-NC-SA-4.0이므로 현재 일반 학습·릴리스 corpus에 섞지 않는다.
- SEVENLLM 및 출처가 합쳐진 instruction dump: 원출처별 라이선스 검토 전까지
  검역한다.
- EMBER2024: 3.2M 정적 feature/label은 LLM SFT 문답이 아니라 별도 악성코드
  classifier 또는 tool-backed 분석용이다. raw malware binary는 수집하지 않는다.
- NIST Juliet/SARD와 PrimeVul: repository/file lineage 분할, 중복 함수 제거와
  안전한 source-only adapter가 완료되기 전까지 확장 SFT에 넣지 않는다.
- 최신 CVE/NVD/KEV/EPSS: 시간에 따라 바뀌는 운영 사실은 weight보다 버전 관리형
  RAG 보안 DB에 둔다.

“최대한 많은 데이터”는 출처·라이선스·안전·오염 게이트를 통과한 범위에서의
최대화를 의미한다. 레코드 수만 늘리기 위해 평가셋, 권리 불명 dump, 악성 바이너리,
자격 증명 또는 피해자 데이터를 학습하지 않는다.
