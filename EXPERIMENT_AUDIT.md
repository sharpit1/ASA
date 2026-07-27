# Experiment Integrity Audit

- Date: 2026-07-24 (Asia/Seoul)
- Scope: `pro6000` five-configuration, ten-image smoke run
- Reviewer: fresh `gpt-5.6-sol` agent, reasoning `ultra`
- Review independence: `same-family`
- Acceptance status: `provisional`
- Overall verdict: **WARN**

## Outcome

The requested smoke scope is operationally complete and its artifact integrity is supported:

- Runs 01, 02, 03, 05, and 06 each completed 10/10 samples.
- All 50 reports record internal attack success.
- All 50 PNG/float32 NPY pairs were directly inspected and are structurally and semantically consistent.
- Run 04 was excluded and no run-04 output was found.
- The actual launcher executes `isolated_vlm_attack/*`; stale repository-root duplicates were not used.
- Flux accepts an absent Hugging Face token and the repaired run completed after an unauthenticated-cache warning.

This supports only the claim “five configurations passed a ten-image pipeline smoke test.” It does not yet support a general ASR, robustness, or method-superiority claim because the standard clean-correct evaluator and an independent ResNet50 replay were not archived.

## Execution chronology

- Scheduled target: 2026-07-24 00:00 KST.
- Orchestrator start: 00:07:31.
- First workload start after preflight/GPU guard: approximately 00:14:47.
- Initial pass: Qwen runs 02 and 03 succeeded; Flux run 01 failed on a required-token check; Bernini runs 05 and 06 failed on the missing/incompatible `decord` dependency.
- Repairs: Flux token handling was made optional and `decord2==3.4.0` was installed.
- Repair completion: 01:40:46, with runs 01, 05, and 06 all exiting zero.

The requested execution was therefore completed, but not at exactly 00:00.

## Artifact audit

For each of the five runs, exactly ten sample directories were present. Across 50 samples:

- PNG: valid RGB PNG, 224×224.
- NPY: `float32`, `(3,224,224)`, C-contiguous, finite, loadable with `allow_pickle=False`.
- PNG bytes equal `round(clip(NPY, 0, 1) * 255)` after CHW→HWC conversion.
- Report: `final_attack_success=true`; recorded adversarial prediction differs from the dataset ground-truth label.
- Summary, report, metrics, query counts, and traces are mutually consistent.
- NPY is saved first, PNG second, and the report later.

Ten NPY files had harmless interpolation overshoot up to `1.00000048`; the NPY preserves classifier input, while the PNG is explicitly clipped and quantized for display.

## Code-path findings

The active callback carries `candidate_classifier_input_float32`. At the first successful victim-query callback, the core:

1. validates and saves the unbatched CHW float32 array;
2. reloads it with `allow_pickle=False` and requires bit-exact equality;
3. saves the clipped/quantized PNG;
4. later writes the report and metrics.

Flux now omits the `token` keyword entirely when the configured token is blank. This works for a public model or a model already available in the local Hugging Face cache. A fresh host accessing an uncached gated repository may still require authentication.

## Integrity classification

- Ground-truth provenance: **PASS**. All 50 summary labels and IDs were checked against `data/nips2017/images.csv`.
- Score normalization: **WARN**. No suspicious self-normalization was found, but runner success is unconditional misclassification rather than clean-correct conditional ASR.
- Result/task fidelity: **PASS for smoke scope**.
- Output plumbing/dead paths: **WARN**. Successful save plumbing is live, but rerun cleanup removes PNG without necessarily removing stale NPY/report/trace files.
- Experimental scope: **WARN**. One dataset, one ResNet50 victim, one seed, ten samples, no repeated trial.
- Evaluation-source classification: **PASS**. Dataset labels are real ground truth; VLM signals are optimization feedback, not evaluation ground truth.

## Remaining risks and recommended follow-up

1. Run `eval/eval_attack_success_float32.py` for every final run and archive clean accuracy, raw numerator/denominator, conditional ASR, and clean/adv predictions.
2. Replay all 50 NPY files through the same pinned ResNet50 revision.
3. Use immutable attempt directories or clear PNG, NPY, reports, metrics, and traces together before a rerun.
4. Preserve a canonical final status plus artifact, dataset, config, model-revision, and code hashes.
5. Treat the no-token Flux result as a warmed-cache/public-access result unless fresh-cache access is separately demonstrated.
6. Resolve or clearly mark the stale repository-root duplicate scripts.
7. Keep any claim limited to five-configuration, ten-image smoke integrity.

Additional operational warning: Bernini populated a redundant central Hugging Face cache, leaving the server disk approximately 95% used with about 82 GB free. No cache was deleted.
