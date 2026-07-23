"""Minimal public surface used by the isolated VLM/AND attack runtimes.

Legacy inversion, fixed-prompt, intermediate-image, and final-rerender entry points are
intentionally not exported and this module has no executable CLI.
"""

from vlm_attack_legacy import (
    PersistentVLMRuntimeCache,
    classifier_input_image,
    evaluate_attack_candidates,
    generate_scene_vocab_words,
    image_to_tensor_01,
    query_vlm_word,
    save_blackbox_prompt_artifacts,
    split_prompt_strip,
)


__all__ = (
    "PersistentVLMRuntimeCache",
    "classifier_input_image",
    "evaluate_attack_candidates",
    "generate_scene_vocab_words",
    "image_to_tensor_01",
    "query_vlm_word",
    "save_blackbox_prompt_artifacts",
    "split_prompt_strip",
)
