# NIPS2017 prompt-corrected DINOv3 / ID-Sim

Prompt-label correction treats a prediction named in the optimized attack prompt as correct and excludes it from attack successes.

| Victim | Before | Excluded | After | ASR before | Corrected ASR | Δ ASR (pp) | DINOv3 sim (mean±std) | ID-Sim sim (mean±std) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| swin | 865 | 13 | 852 | 90.48% | 89.12% | -1.36 pp | 0.708054±0.210634 | 0.671450±0.181499 |
| vim-small | 866 | 10 | 856 | 92.62% | 91.55% | -1.07 pp | 0.716386±0.200001 | 0.681212±0.175895 |

The raw audit CSV records the source prompt, matched category labels, and exclusion decision for every previously selected sample.
