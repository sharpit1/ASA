# NIPS2017 ablation_and_no_verifier: DINOv3 / ID-Sim

## Evaluation scope

- Primary selection: `clean_pred == true_label` and stored `final_attack_success == true`.
- Reference: the original `data/nips2017/images/<ImageId>.png`.
- Attack: each sample's persisted `images/final_selected.png`.
- DINOv3: ViT-L/16 LVD-1689M final normalized CLS cosine similarity.
- ID-Sim: official DINOv3 ViT-L/16 checkpoint, PIL preprocessing, and CLS mode.
- Similarity is higher-is-better; distance is `1 - similarity` and lower-is-better.

## Raw summary

| Victim | Clean-correct | Selected | PNG-retained | DINOv3 sim (mean±std) | ID-Sim sim (mean±std) |
|---|---:|---:|---:|---:|---:|
| swin | 956 | 865 | 865 | 0.703785±0.214032 | 0.669212±0.183216 |
| vim-small | 935 | 866 | 866 | 0.711798±0.205314 | 0.677979±0.178759 |

## Verification notes

- `PNG-retained` is the subset that remains adversarial after reloading the persisted 8-bit PNG and rerunning the victim model.
- The JSON summary contains median, quartiles, minimum, maximum, and separate aggregates for the PNG-retained subset.
- No score is imputed for missing or non-finite samples.
