# SigLIP2·CLIP query-budget ASR 분석

## 산출 기준

- 분모는 victim별 clean-correct 샘플(SigLIP2 859개, CLIP 778개)이다.
- 성공 query는 각 `report.json`의 실제 종료 시점 `victim_query_count`를 사용한다.
- classifier 공격 성공이어도 naturalness가 false이면 실패로 처리하고 유효 query를 100으로 둔다.
- 나머지 실패도 유효 query를 100으로 둔다.
- CLIP은 in-loop naturalness 결과, SigLIP2는 사후 Gemma 검증 결과를 사용한다.

## Query 제한별 ASR

| Victim | Clean-correct | Q20 | Q40 | Q60 | Q80 | Q100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SigLIP2 | 859 | 491 (57.16%) | 565 (65.77%) | 588 (68.45%) | 594 (69.15%) | 602 (70.08%) |
| CLIP ViT-B/16 | 778 | 570 (73.26%) | 636 (81.75%) | 655 (84.19%) | 661 (84.96%) | 667 (85.73%) |

## Query 통계

| Victim | Classifier success 관측 | Naturalness로 최종 거절 | 최종 성공 | 최종 실패 | 성공만 Avg Q | 전체 capped Avg Q | 실패 Avg Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SigLIP2 | 615 | 13 | 602 | 257 | 13.12 | 39.11 | 100.00 |
| CLIP ViT-B/16 | 668 | 1 | 667 | 111 | 11.41 | 24.05 | 100.00 |

## Query 100에서 다른 공격과 비교

| Victim | Attack | 성공 / Attacked | ASR | 전체 capped Avg Q | 성공만 Avg Q | ASA 대비 ASR 차이 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SigLIP2 | ASA (본 결과) | 602 / 859 | 70.08% | 39.11 | 13.12 | +0.00%p |
| SigLIP2 | AdvFlow | 113 / 859 | 13.15% | 89.41 | 19.47 | -56.93%p |
| SigLIP2 | DIFAttack | 301 / 859 | 35.04% | 78.63 | 39.02 | -35.04%p |
| SigLIP2 | MCGAttack | 377 / 859 | 43.89% | 64.16 | 18.35 | -26.19%p |
| SigLIP2 | CGAttack | 236 / 859 | 27.47% | 79.31 | 24.68 | -42.61%p |
| CLIP ViT-B/16 | ASA (본 결과) | 667 / 778 | 85.73% | 24.05 | 11.41 | +0.00%p |
| CLIP ViT-B/16 | AdvFlow | 175 / 778 | 22.49% | 82.80 | 23.54 | -63.24%p |
| CLIP ViT-B/16 | DIFAttack | 356 / 778 | 45.76% | 71.56 | 37.84 | -39.97%p |
| CLIP ViT-B/16 | MCGAttack | 445 / 778 | 57.20% | 51.93 | 15.96 | -28.53%p |
| CLIP ViT-B/16 | CGAttack | 252 / 778 | 32.39% | 76.24 | 26.65 | -53.34%p |

## 비교 해석상 제한

- 다른 네 공격은 문서상 classifier 성공 기준이며 naturalness verifier가 적용되지 않았다.
- 다른 네 공격은 `epsilon=12/255` 픽셀 제약 공격이고 ASA는 생성·프롬프트 기반 공격이므로 ASR만으로 직접적인 우열을 확정할 수 없다.
- SigLIP2 naturalness는 저장된 224×224 `attack_success.png`에 대한 사후 검증이다. 실행 중 full-resolution 후보를 검증한 CLIP과 입력이 완전히 같지는 않다.
- 제공 문서에는 다른 공격의 개별 `progress.csv`가 없어 Q20/Q40/Q60/Q80 ASR은 재계산할 수 없다. 따라서 다른 공격 비교는 Q100에 한정한다.
