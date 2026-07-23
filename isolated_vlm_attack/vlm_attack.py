"""Minimal VLM helpers for the isolated robustness-attack pipeline.

This module deliberately excludes inversion, legacy CLI flows, and subprocess
render fallbacks.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from vlm_runtime import infer_vlm_backend, load_vlm_runtime
PersistentFluxRenderSession = Any
TorchvisionClassifier = Any
STOPWORDS = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'for', 'with', 'and', 'to', 'is', 'are', 'this', 'that', 'it', 'image', 'photo', 'picture', 'scene', 'background'}
_CUDNN_SDPA_VIM_SMALL_CONFIGURED: Optional[bool] = None

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if text in {'1', 'true', 'yes', 'on'}:
        return True
    if text in {'0', 'false', 'no', 'off'}:
        return False
    return bool(default)

def _is_vim_small_classifier(name: object) -> bool:
    text = str(name or '').strip().lower().replace('_', '-')
    return text in {'vim-small', 'mambavision'}

def _maybe_disable_cudnn_sdpa_for_vim_small(classifier_name: object, *, context: str) -> None:
    global _CUDNN_SDPA_VIM_SMALL_CONFIGURED
    if not _is_vim_small_classifier(classifier_name):
        return
    if _CUDNN_SDPA_VIM_SMALL_CONFIGURED is True:
        return
    backend = getattr(getattr(torch, 'backends', None), 'cuda', None)
    if backend is None or not torch.cuda.is_available():
        return
    try:
        disable_cudnn = not _env_bool('GCG_ENABLE_CUDNN_SDPA', False) and _env_bool('GCG_VIM_SMALL_DISABLE_CUDNN_SDPA', True) and hasattr(backend, 'enable_cudnn_sdp')
        prefer_flash = _env_bool('GCG_VIM_SMALL_PREFER_FLASH_SDPA', True)
        force_flash = _env_bool('GCG_VIM_SMALL_FORCE_FLASH_SDPA', False)
        if prefer_flash and hasattr(backend, 'enable_flash_sdp'):
            backend.enable_flash_sdp(True)
        if force_flash:
            if hasattr(backend, 'enable_mem_efficient_sdp'):
                backend.enable_mem_efficient_sdp(False)
            if hasattr(backend, 'enable_math_sdp'):
                backend.enable_math_sdp(False)
        if disable_cudnn:
            cudnn_enabled = True
            if hasattr(backend, 'cudnn_sdp_enabled'):
                cudnn_enabled = bool(backend.cudnn_sdp_enabled())
            if cudnn_enabled:
                backend.enable_cudnn_sdp(False)
        flash_state = bool(backend.flash_sdp_enabled()) if hasattr(backend, 'flash_sdp_enabled') else None
        mem_state = bool(backend.mem_efficient_sdp_enabled()) if hasattr(backend, 'mem_efficient_sdp_enabled') else None
        math_state = bool(backend.math_sdp_enabled()) if hasattr(backend, 'math_sdp_enabled') else None
        cudnn_state = bool(backend.cudnn_sdp_enabled()) if hasattr(backend, 'cudnn_sdp_enabled') else None
        print(f"[vlm_attack] configured SDPA for vim-small/mambavision (context='{context}', flash={flash_state}, mem_efficient={mem_state}, math={math_state}, cudnn={cudnn_state}, force_flash={force_flash}).")
        _CUDNN_SDPA_VIM_SMALL_CONFIGURED = True
    except Exception as exc:
        print(f"[vlm_attack] WARNING: failed to configure SDPA for vim-small (context='{context}'; {type(exc).__name__}: {exc})")
        _CUDNN_SDPA_VIM_SMALL_CONFIGURED = False

class PersistentVLMRuntimeCache:

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str, str], Dict[str, object]] = {}

    def query(self, *, image_path: Path, question: str, vlm_backend: str, vlm_model_id: str, vlm_device_raw: str, max_new_tokens: int, enable_thinking: bool, do_sample: bool) -> Tuple[str, Optional[str]]:
        device = resolve_vlm_device(vlm_device_raw)
        backend = infer_vlm_backend(vlm_backend=str(vlm_backend), model_id=str(vlm_model_id), allow_blip=True)
        cache_key = (backend, str(vlm_model_id), str(device))
        runtime = self._cache.get(cache_key)
        if runtime is None:
            dtype = torch.float16 if device.type == 'cuda' else torch.float32
            vlm_model, vlm_processor, ask_fn, uses_pipeline_backend = load_vlm_runtime(backend=backend, model_id=str(vlm_model_id), vlm_dtype=dtype, vlm_device=device, allow_blip=True)
            if not uses_pipeline_backend and hasattr(vlm_model, 'to'):
                vlm_model = vlm_model.to(device)
                if hasattr(vlm_model, 'eval'):
                    vlm_model.eval()
            runtime = {'model': vlm_model, 'processor': vlm_processor, 'ask_fn': ask_fn, 'uses_pipeline_backend': bool(uses_pipeline_backend), 'device': device}
            self._cache[cache_key] = runtime
        try:
            image = Image.open(image_path).convert('RGB')
            with torch.no_grad():
                raw_answer = runtime['ask_fn'](image=image, question=str(question), model=runtime['model'], processor=runtime['processor'], device=runtime['device'], max_new_tokens=int(max_new_tokens), enable_thinking=bool(enable_thinking), do_sample=bool(do_sample))
            return (str(raw_answer or '').strip(), None)
        except Exception as exc:
            return ('', str(exc))

    def close(self) -> None:
        for runtime in self._cache.values():
            model = runtime.get('model')
            uses_pipeline_backend = bool(runtime.get('uses_pipeline_backend', False))
            if model is not None and hasattr(model, 'to') and (not uses_pipeline_backend):
                try:
                    model.to('cpu')
                except Exception:
                    pass
        self._cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

def resolve_vlm_device(raw: str) -> torch.device:
    token = str(raw or '').strip().lower()
    if token in {'', 'auto'}:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(token)

def is_flux2_klein_model_path(model_path: object) -> bool:
    token = str(model_path or '').strip().lower()
    if not token:
        return False
    return ('flux.2' in token or 'flux2' in token) and 'klein' in token

def normalize_text(text: str) -> str:
    text = str(text or '').strip().lower()
    text = re.sub('[^a-z0-9_\\-\\s]', ' ', text)
    text = re.sub('\\s+', ' ', text).strip()
    return text

def pick_generic_word(answer: str, extra_stopwords: Optional[set[str]]=None) -> Optional[str]:
    stopwords = STOPWORDS if not extra_stopwords else STOPWORDS.union(extra_stopwords)
    for token in answer.replace('-', ' ').split():
        if len(token) >= 3 and token not in stopwords:
            return token
    return None

def normalize_scene(raw_answer: str, fallback: str) -> str:
    answer = normalize_text(raw_answer)
    if not answer:
        return fallback
    for key, value in [('indoor', 'indoor'), ('inside', 'indoor'), ('outdoor', 'outdoor'), ('beach', 'beach'), ('forest', 'forest'), ('mountain', 'mountain'), ('street', 'street'), ('city', 'city'), ('park', 'park'), ('field', 'field'), ('desert', 'desert'), ('snow', 'snow'), ('night', 'night')]:
        if key in answer:
            return value
    return pick_generic_word(answer) or fallback

def normalize_object(raw_answer: str, fallback: str) -> str:
    answer = normalize_text(raw_answer)
    if not answer:
        return fallback
    return pick_generic_word(answer, extra_stopwords={'main', 'subject', 'next', 'nearby', 'adjacent', 'beside', 'near'}) or fallback

def normalize_slot_value(raw_answer: str, fallback: str, slot_kind: str) -> str:
    if slot_kind == 'scene':
        return normalize_scene(raw_answer, fallback)
    return normalize_object(raw_answer, fallback)

def _strip_balanced_quotes(text: str) -> str:
    out = str(text or '').strip()
    if len(out) >= 2 and out[0] in {"'", '"', '`'} and (out[-1] == out[0]):
        out = out[1:-1].strip()
    return out

def normalize_scene_vocab_word(text: str) -> str:
    out = str(text or '').strip()
    out = re.sub('^\\s*(?:[-*]|\\d+[\\.\\)])\\s*', '', out)
    out = _strip_balanced_quotes(out)
    out = re.sub('\\s+', ' ', out).strip()
    out = out.strip(' ,.;:!?')
    return out

def slot_prompt_spec(slot_kind: str, max_words: int=5) -> Dict[str, str]:
    candidate_length_text = '1 word' if int(max_words) == 1 else f'1~{int(max_words)} words'
    if slot_kind == 'scene':
        return {'kind': 'scene', 'marker': '<scene>', 'anchor_label': 'Scene/topic anchor', 'candidate_requirement': f'- Each candidate must be a concise English phrase of {candidate_length_text}.\n', 'relevance_requirement': '- Candidates should stay visually meaningful in relation to the current prompt context.\n'}
    return {'kind': 'object', 'marker': '<object>', 'anchor_label': 'Object/topic anchor', 'candidate_requirement': f'- Each candidate must be a concise English physical-object noun or phrase of {candidate_length_text}.\n', 'relevance_requirement': '- Candidates should stay visually meaningful in relation to the current prompt context.\n'}

def scene_vocab_strategy_specs() -> List[Dict[str, str]]:
    return [{'name': 'background_shift', 'title': 'Background Shift', 'description': 'Commands that modify or replace the surrounding environment.'}, {'name': 'weather_atmosphere', 'title': 'Weather & Atmosphere', 'description': 'Commands that introduce weather, haze, lighting, or atmospheric effects.'}, {'name': 'texture_material', 'title': 'Texture & Material', 'description': 'Commands that alter surface properties, materials, or texture cues.'}]

def _match_scene_vocab_strategy_name(raw: object) -> Optional[str]:
    token = normalize_text(str(raw or ''))
    if not token:
        return None
    alias_map = {'background': 'background_shift', 'background shift': 'background_shift', 'background_shift': 'background_shift', 'environment': 'background_shift', 'environment shift': 'background_shift', 'weather': 'weather_atmosphere', 'atmosphere': 'weather_atmosphere', 'weather atmosphere': 'weather_atmosphere', 'weather_atmosphere': 'weather_atmosphere', 'atmospheric': 'weather_atmosphere', 'texture': 'texture_material', 'material': 'texture_material', 'texture material': 'texture_material', 'texture_material': 'texture_material', 'surface': 'texture_material'}
    if token in alias_map:
        return alias_map[token]
    return None

def resolve_scene_vocab_strategy_specs(raw: object='all') -> List[Dict[str, str]]:
    specs = scene_vocab_strategy_specs()
    if raw is None:
        return list(specs)
    if isinstance(raw, (list, tuple, set)):
        tokens: List[str] = []
        for item in raw:
            tokens.extend(re.split('[,;|]+', str(item or '')))
    else:
        text = str(raw).strip()
        if not text:
            return []
        token = text.lower()
        if token in {'all', 'default', '1', 'true', 'yes', 'on'}:
            return list(specs)
        if token in {'none', 'off', '0', 'false', 'no', 'disable', 'disabled'}:
            return []
        tokens = re.split('[,;|]+', text)
    selected: List[Dict[str, str]] = []
    selected_names = set()
    for raw_token in tokens:
        token = str(raw_token or '').strip()
        if not token:
            continue
        token_l = token.lower()
        if token_l in {'all', 'default', '1', 'true', 'yes', 'on'}:
            return list(specs)
        if token_l in {'none', 'off', '0', 'false', 'no', 'disable', 'disabled'}:
            continue
        name = _match_scene_vocab_strategy_name(token)
        if name is None:
            valid = ', '.join((str(item['name']) for item in specs))
            raise ValueError(f"--gcg_scene_vocab_enabled_strategies contains unknown strategy '{token}'. Valid values are: all, none, {valid}.")
        if name in selected_names:
            continue
        selected_names.add(name)
        for spec in specs:
            if str(spec['name']) == name:
                selected.append(dict(spec))
                break
    return selected

def format_scene_vocab_strategy_lines(strategy_specs: Sequence[Dict[str, str]], *, step_idx: int) -> str:
    initial_lines = {'background_shift': "Commands that entirely replace or subtly alter the surrounding environment (e.g., 'move the subject to a dusty warehouse', 'place the scene in a dense forest').", 'weather_atmosphere': "Commands that introduce meteorological changes or atmospheric conditions affecting global lighting (e.g., 'add a thick morning fog', 'turn the weather into a torrential downpour').", 'texture_material': "Alter ONLY the surface material, texture, or color of the target object. The object's structural shape and core identity MUST remain 100% intact. Apply contradictory or highly detailed surface wraps using modifiers like 'coated in', 'wrapped in', or 'painted with' (e.g., 'coated in iridescent bioluminescent scales', 'wrapped in Vantablack velvet', 'painted with glitching neon chrome'). Do NOT alter the photo medium itself (no film grain, no scratched lenses) and do NOT change what the object fundamentally is."}
    iterative_lines = {'background_shift': 'Generate commands that modify the environment. If previous background shifts failed, try contrasting indoor/outdoor settings or changing the era/time period.', 'weather_atmosphere': 'Generate commands introducing atmospheric effects. Focus on conditions that logically conflict with the original lighting (e.g., adding heavy rain to a sunny scene).', 'texture_material': "Alter ONLY the surface material, texture, or color of the target object. The object's structural shape and core identity MUST remain 100% intact. Apply contradictory or highly detailed surface wraps using modifiers like 'coated in', 'wrapped in', or 'painted with' (e.g., 'coated in iridescent bioluminescent scales', 'wrapped in Vantablack velvet', 'painted with glitching neon chrome'). Do NOT alter the photo medium itself (no film grain, no scratched lenses) and do NOT change what the object fundamentally is."}
    line_map = initial_lines if int(step_idx) == 0 else iterative_lines
    lines: List[str] = []
    for idx, spec in enumerate(strategy_specs, start=1):
        name = str(spec.get('name', '')).strip()
        title = str(spec.get('title', name)).strip()
        description = line_map.get(name, str(spec.get('description', '')).strip())
        lines.append(f'  * Strategy {int(idx)} ({title}): {description}\n')
    return ''.join(lines)

def format_flux2_attribute_constraint(strategy_specs: Sequence[Dict[str, str]]) -> str:
    enabled_terms = set()
    for spec in strategy_specs:
        name = str(spec.get('name', '')).strip()
        if name == 'background_shift':
            enabled_terms.add('background')
        elif name == 'weather_atmosphere':
            enabled_terms.add('weather')
        elif name == 'texture_material':
            enabled_terms.update(['color', 'texture'])
    terms = [term for term in ('background', 'color', 'texture', 'weather') if term in enabled_terms]
    if len(terms) == 0:
        return 'We constrain prompts to attribute-level edits. Do Not use Prompts that instantiate a new named entity, new category, or class-specific material. This preserves the attack mechanism while preventing explicit class-concept injection. \n'
    if len(terms) == 1:
        term_list = terms[0]
    elif len(terms) == 2:
        term_list = ' and '.join(terms)
    else:
        term_list = ', '.join(terms[:-1]) + f', and {terms[-1]}'
    return f'We allow verb-driven transformations of {term_list}, but constrain them to attribute-level edits. Do Not use Prompts that instantiate a named object, location, scene category, or class-specific material. This preserves the attack mechanism while preventing explicit class-concept injection. \n'

def extract_json_payload(raw: str) -> Optional[Any]:
    text = str(raw or '').strip()
    if not text:
        return None
    decoder = json.JSONDecoder()
    parsed_candidates: List[Any] = []
    for idx, ch in enumerate(text):
        if ch not in '[{':
            continue
        try:
            payload, _ = decoder.raw_decode(text[idx:])
        except Exception:
            continue
        parsed_candidates.append(payload)

    def looks_like_candidate_payload(payload: Any) -> bool:
        if isinstance(payload, list):
            return True
        if isinstance(payload, dict):
            if isinstance(payload.get('strategies'), list):
                return True
            for key in ('candidates', 'object_words', 'objects', 'scene_words', 'words', 'vocab'):
                if isinstance(payload.get(key), list):
                    return True
        return False

    def looks_like_strategy_group_payload(payload: Any) -> bool:
        return isinstance(payload, dict) and isinstance(payload.get('strategies'), list)
    for payload in reversed(parsed_candidates):
        if looks_like_strategy_group_payload(payload):
            return payload
    for payload in reversed(parsed_candidates):
        if looks_like_candidate_payload(payload):
            return payload
    for payload in reversed(parsed_candidates):
        if isinstance(payload, (dict, list)):
            return payload
    return None

def _normalize_feedback_note(raw: object) -> str:
    text = str(raw or '').strip()
    if not text:
        return ''
    text = re.sub('\\s+', ' ', text).strip()
    return text.strip(' ,.;:!?')

def parse_scene_vocab_words(raw_answer: str, limit: int) -> List[str]:
    payload = extract_json_payload(raw_answer)
    raw_items: Optional[Sequence[object]] = None
    if isinstance(payload, dict):
        for key in ('candidates', 'object_words', 'objects', 'scene_words', 'words', 'vocab'):
            value = payload.get(key)
            if isinstance(value, list):
                raw_items = value
                break
    elif isinstance(payload, list):
        raw_items = payload
    if raw_items is None:
        quoted_tokens = re.findall('[\\"\']([^\\"\']+)[\\"\']', str(raw_answer or ''))
        if len(quoted_tokens) > 0:
            raw_items = quoted_tokens
        else:
            raw_items = re.split('[\\n,;]+', str(raw_answer or ''))
    words: List[str] = []
    seen = set()
    parser_meta_tokens = {'json', 'candidates', 'object_words', 'scene_words', 'words', 'vocab'}
    for item in raw_items:
        candidate = item
        if isinstance(item, dict):
            for key in ('word', 'candidate', 'text', 'object', 'object_word', 'scene_word'):
                if key in item:
                    candidate = item[key]
                    break
        normalized = normalize_scene_vocab_word(str(candidate))
        if len(normalized) == 0:
            continue
        key = normalized.lower()
        if key in parser_meta_tokens:
            continue
        if key in seen:
            continue
        seen.add(key)
        words.append(normalized)
        if len(words) >= int(limit):
            break
    return words

def parse_scene_vocab_strategy_groups(raw_answer: str, *, prompts_per_strategy: int, strategy_specs: Optional[Sequence[Dict[str, str]]]=None) -> List[Dict[str, object]]:
    specs = list(strategy_specs) if strategy_specs is not None else scene_vocab_strategy_specs()
    if len(specs) == 0:
        return []
    spec_by_name = {str(item['name']): item for item in specs}
    payload = extract_json_payload(raw_answer)
    groups_by_name: Dict[str, List[str]] = {str(item['name']): [] for item in specs}

    def _normalize_candidates(raw_candidates: object) -> List[str]:
        if not isinstance(raw_candidates, list):
            return []
        words: List[str] = []
        seen = set()
        for item in raw_candidates:
            candidate = item
            if isinstance(item, dict):
                for key in ('word', 'candidate', 'text', 'object', 'object_word', 'scene_word'):
                    if key in item:
                        candidate = item[key]
                        break
            normalized = normalize_scene_vocab_word(str(candidate))
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            words.append(normalized)
            if len(words) >= int(prompts_per_strategy):
                break
        return words
    if isinstance(payload, dict):
        raw_strategies = payload.get('strategies')
        if isinstance(raw_strategies, list):
            for item in raw_strategies:
                if not isinstance(item, dict):
                    continue
                strategy_name = _match_scene_vocab_strategy_name(item.get('name', item.get('strategy', item.get('title'))))
                if strategy_name is None or strategy_name not in spec_by_name:
                    continue
                groups_by_name[strategy_name] = _normalize_candidates(item.get('candidates'))
        for spec in specs:
            strategy_name = str(spec['name'])
            if len(groups_by_name[strategy_name]) > 0:
                continue
            raw_candidates = payload.get(strategy_name)
            if raw_candidates is None:
                raw_candidates = payload.get(str(spec['title']))
            if raw_candidates is None:
                raw_candidates = payload.get(str(spec['title']).lower())
            if raw_candidates is None:
                continue
            groups_by_name[strategy_name] = _normalize_candidates(raw_candidates)
    groups: List[Dict[str, object]] = []
    for spec in specs:
        strategy_name = str(spec['name'])
        candidates = groups_by_name[strategy_name]
        groups.append({'name': strategy_name, 'title': str(spec['title']), 'candidates': list(candidates)})
    if any((len(group['candidates']) > 0 for group in groups)):
        return groups
    flat_words = parse_scene_vocab_words(str(raw_answer or ''), limit=max(1, int(prompts_per_strategy)) * len(specs))
    if len(flat_words) == 0:
        return []
    groups = []
    cursor = 0
    for spec in specs:
        next_cursor = cursor + max(1, int(prompts_per_strategy))
        groups.append({'name': str(spec['name']), 'title': str(spec['title']), 'candidates': list(flat_words[cursor:next_cursor])})
        cursor = next_cursor
    return groups

def flatten_scene_vocab_strategy_groups(groups: Sequence[Dict[str, object]]) -> List[Dict[str, str]]:
    normalized_groups: List[Dict[str, object]] = []
    for item in groups:
        if not isinstance(item, dict):
            continue
        strategy_name = str(item.get('name', '')).strip()
        strategy_title = str(item.get('title', strategy_name)).strip()
        raw_candidates = item.get('candidates')
        if not strategy_name or not isinstance(raw_candidates, list):
            continue
        normalized_groups.append({'name': strategy_name, 'title': strategy_title, 'candidates': [str(candidate).strip() for candidate in raw_candidates if str(candidate).strip()]})
    if len(normalized_groups) == 0:
        return []
    entries: List[Dict[str, str]] = []
    max_len = max((len(item['candidates']) for item in normalized_groups))
    for candidate_idx in range(int(max_len)):
        for item in normalized_groups:
            candidates = item['candidates']
            if candidate_idx >= len(candidates):
                continue
            entries.append({'strategy_name': str(item['name']), 'strategy_title': str(item['title']), 'word': str(candidates[candidate_idx])})
    return entries

def format_scene_vocab_feedback(*, feedback_entries: Sequence[Dict[str, object]], enabled: bool, limit: int) -> str:
    if not enabled or len(feedback_entries) == 0:
        return 'No previous-step feedback is available yet.'
    k = min(max(1, int(limit)), len(feedback_entries))
    ranked = sorted(feedback_entries, key=scene_feedback_sort_key, reverse=True)[:k]
    lines = ['Previously tried candidates and best objectives:']
    for item in ranked:
        word = str(item.get('scene_word', '')).strip()
        if len(word) == 0:
            continue
        objective = scene_feedback_objective(item)
        attempts = max(1, int(item.get('attempts', 1)))
        naturalness_note = _normalize_feedback_note(item.get('naturalness_feedback'))
        naturalness_is_natural = item.get('naturalness_is_natural')
        if objective is not None:
            line = f'- {word} | best_objective={objective + 10.0:.6f} | attempts={attempts}'
        else:
            line = f'- {word} | best_objective=unscored | attempts={attempts}'
        if naturalness_is_natural is False:
            line += ' | naturalness=unnatural'
        if naturalness_note:
            line += f' | evaluator_feedback={naturalness_note}'
        lines.append(line)
    if len(lines) == 1:
        return 'No previous-step feedback is available yet.'
    return '\n'.join(lines)

def scene_feedback_objective(item: Dict[str, object]) -> Optional[float]:
    raw = item.get('objective')
    if raw is None:
        return None
    try:
        objective = float(raw)
    except Exception:
        return None
    if not np.isfinite(objective):
        return None
    return objective

def scene_feedback_sort_key(item: Dict[str, object]) -> Tuple[float, int, str]:
    objective = scene_feedback_objective(item)
    objective_rank = objective if objective is not None else float('-inf')
    return (objective_rank, int(item.get('attempts', 0)), str(item.get('scene_word', '')))

def build_marked_prompt(current_prompt: str, current_word: str, occurrence: int, marker: str) -> str:
    text = str(current_prompt or '')
    word = str(current_word or '').strip()
    if word:
        matches = list(re.finditer(f'\\b{re.escape(word)}\\b', text, flags=re.IGNORECASE))
        if 0 <= int(occurrence) < len(matches):
            match = matches[int(occurrence)]
            return text[:match.start()] + marker + text[match.end():]
    for slot_marker in ('<scene>', '{scene}', '<object>', '{object}'):
        if slot_marker in text:
            return text.replace(slot_marker, marker, 1)
    return text

def generate_scene_vocab_words(*, args: argparse.Namespace, step_idx: int, current_prompt: str, current_word: str, slot_kind: str, best_objective: float, previous_feedback: Sequence[Dict[str, object]], reference_image_path: Path, fallback_word: str, runtime_cache: Optional[PersistentVLMRuntimeCache]=None) -> Tuple[List[str], str, str, Optional[str]]:
    _maybe_disable_cudnn_sdpa_for_vim_small(getattr(args, 'classifier_name', None), context='generate_scene_vocab_words')
    setattr(args, '_scene_vocab_strategy_groups', [])
    setattr(args, '_scene_vocab_strategy_entries', [])
    slot_candidate_max_words = max(1, int(getattr(args, 'gcg_slot_candidate_max_words', 5)))
    prompts_per_strategy = max(0, int(getattr(args, 'gcg_scene_vocab_prompts_per_strategy', 0)))
    strategy_specs = resolve_scene_vocab_strategy_specs(getattr(args, 'gcg_scene_vocab_enabled_strategies', 'all'))
    strategy_prompt_model_enabled = bool(is_flux2_klein_model_path(getattr(args, 'model_path', None)) or bool(getattr(args, 'bernini_edit_prompt_mode', False)) or bool(getattr(args, 'qwen_edit_prompt_mode', False)) or bool(getattr(args, 'qwen_strategy_and_enable', False)))
    flux2_strategy_prompt_mode = bool(prompts_per_strategy > 0 and len(strategy_specs) > 0 and strategy_prompt_model_enabled)
    all_strategy_specs = scene_vocab_strategy_specs()
    using_default_strategy_set = [str(item['name']) for item in strategy_specs] == [str(item['name']) for item in all_strategy_specs]
    slot_spec = slot_prompt_spec(slot_kind, max_words=slot_candidate_max_words)
    slot_marker = slot_spec['marker']
    marked_prompt = build_marked_prompt(current_prompt, current_word, int(args.gcg_occurrence), slot_marker)
    current_value = normalize_scene_vocab_word(str(current_word))
    if not current_value:
        current_value = normalize_scene_vocab_word(str(fallback_word))
    slot_topic = str(args.gcg_scene_vocab_topic or current_value).strip()
    if not slot_topic:
        slot_topic = fallback_word
    target_class_name = str(args.class_name or '').strip() or None
    class_requirement = f'- Every candidate must be semantically related to the target class "<class>" (for this example: "{target_class_name}").\n' if target_class_name else '- Every candidate must be semantically related to the target class "<class>".\n'
    has_naturalness_feedback = any((item.get('naturalness_is_natural') is False or bool(_normalize_feedback_note(item.get('naturalness_feedback'))) for item in previous_feedback))
    feedback_block = format_scene_vocab_feedback(feedback_entries=previous_feedback, enabled=bool(args.gcg_scene_vocab_feedback), limit=int(args.gcg_scene_feedback_limit))
    visual_feedback_block = 'Adversarial visual feedback:\n- The reference image is attached and represents the most recent generated result.\n'
    cwor_instruction_block = ''
    if bool(getattr(args, 'cwor_enable', False)) and (not flux2_strategy_prompt_mode):
        cwor_instruction_block = "- EXCEPTION (OVERRIDE): If '<CWOR>' is identified as the best candidate from the previous step, STRICTLY IGNORE Strategy 1. Instead, use Strategy 2 exclusively to generate exactly 2 novel words that are completely unrelated to '<CWOR>'.\n- Crucially, analyze ALL previous-step candidates (both successes and failures), EXCLUDING '<CWOR>':\n  * STRICTLY IGNORE '<CWOR>' during your analysis. Do not extract or learn any traits from it.\n  * Winners: Extract successful semantic/visual traits from the REMAINING best candidates to fuel Strategy 1 (unless the override applies).\n"
    naturalness_feedback_instruction = "- If a feedback entry says 'naturalness=unnatural', treat it as a warning and avoid reusing the artifact pattern described in its evaluator_feedback.\n" if has_naturalness_feedback else ''
    if len(strategy_specs) == 0:
        strategy_instruction = ''
    elif using_default_strategy_set:
        if step_idx == 0:
            strategy_instruction = "- Since this is the initial step (no previous feedback), establish a baseline by equally distributing your candidates across THREE distinct attack vectors:\n  * Strategy 1 (Background Shift): Commands that entirely replace or subtly alter the surrounding environment (e.g., 'move the subject to a dusty warehouse', 'place the scene in a dense forest').\n  * Strategy 2 (Weather & Atmosphere): Commands that introduce meteorological changes or atmospheric conditions affecting global lighting (e.g., 'add a thick morning fog', 'turn the weather into a torrential downpour').\n  * Strategy 3 (Material & Color Reskinning): Alter ONLY the surface material, texture, or color of the target object. The object's structural shape and core identity MUST remain 100% intact. Apply contradictory or highly detailed surface wraps. CRITICAL SYNTAX: You MUST start your candidate with a past participle (e.g., 'coated in', 'wrapped in', 'painted with', 'covered in'). Do NOT start with a noun or a base verb (e.g., NEVER output 'coat in dark bronze', strictly output 'coated in dark bronze'). Do NOT alter the photo medium itself and do NOT change what the object fundamentally is.\n- Propose diverse concepts within these three categories to discover which dimension the target model is most sensitive to.\n"
        else:
            strategy_instruction = f"- Employ the following THREE distinct generation strategies to systematically test the model's vulnerabilities. Use the feedback to refine your attacks:\n  * Strategy 1 (Background Shift): Generate commands that modify the environment. If previous background shifts failed, try contrasting indoor/outdoor settings or changing the era/time period.\n  * Strategy 2 (Weather & Atmosphere): Generate commands introducing atmospheric effects. Focus on conditions that logically conflict with the original lighting (e.g., adding heavy rain to a sunny scene).\n  * Strategy 3 (Material & Color Reskinning): Alter ONLY the surface material, texture, or color of the target object. The object's structural shape and core identity MUST remain 100% intact. Apply contradictory or highly detailed surface wraps. CRITICAL SYNTAX: You MUST start your candidate with a past participle (e.g., 'coated in', 'wrapped in', 'painted with', 'covered in'). Do NOT start with a noun or a base verb (e.g., NEVER output 'coat in dark bronze', strictly output 'coated in dark bronze'). Do NOT alter the photo medium itself and do NOT change what the object fundamentally is.\n{cwor_instruction_block}  * Exploit Winners: Identify which of the three categories (Background, Weather, Texture) produced the best candidate in the feedback. Generate more compounded variants within that winning category.\n  * Avoid Losers: Identify the underlying characteristics of the remaining ineffective candidates. STRICTLY AVOID generating new commands that share these traits or fall into the weakest category.\n"
    else:
        strategy_count = int(len(strategy_specs))
        strategy_lines = format_scene_vocab_strategy_lines(strategy_specs, step_idx=int(step_idx))
        if step_idx == 0:
            strategy_instruction = f'- Since this is the initial step (no previous feedback), establish a baseline using the enabled {strategy_count} strategy bucket(s):\n{strategy_lines}- Propose diverse concepts within the enabled categories to discover which dimension the target model is most sensitive to.\n'
        else:
            strategy_instruction = f"- Employ the enabled {strategy_count} generation strategy bucket(s) to test the model's vulnerabilities. Use the feedback to refine your attacks:\n{strategy_lines}{cwor_instruction_block}  * Exploit Winners: Identify which enabled category produced the best candidate in the feedback and generate stronger variants within it.\n  * Avoid Losers: Identify ineffective traits in the remaining candidates. STRICTLY AVOID generating new commands that share these traits or fall into disabled or weak categories.\n"
    if strategy_prompt_model_enabled:
        if flux2_strategy_prompt_mode:
            if using_default_strategy_set:
                strategy_instruction = f'{strategy_instruction}- Keep the candidates separated by strategy in the final JSON output.\n- For EACH strategy, generate exactly {int(prompts_per_strategy)} unique candidates.\n- Keep the three strategy buckets balanced. Do not move candidates between buckets.\n'
            else:
                strategy_instruction = f'{strategy_instruction}- Keep the candidates separated by enabled strategy in the final JSON output.\n- For EACH enabled strategy, generate exactly {int(prompts_per_strategy)} unique candidates.\n- Do not invent disabled strategy buckets or move candidates between buckets.\n'
            total_candidate_count = int(prompts_per_strategy) * int(len(strategy_specs))
            if using_default_strategy_set:
                output_schema_line = '- Output ONLY valid JSON with the schema {"strategies": [{"name": "background_shift", "candidates": ["...", "..."]}, {"name": "weather_atmosphere", "candidates": ["...", "..."]}, {"name": "texture_material", "candidates": ["...", "..."]}]}.\n'
            else:
                schema_parts = ['{"name": "' + str(spec['name']) + '", "candidates": ["...", "..."]}' for spec in strategy_specs]
                output_schema_line = '- Output ONLY valid JSON with the schema ' + '{"strategies": [' + ', '.join(schema_parts) + ']}.\n'
        else:
            total_candidate_count = int(args.gcg_scene_vocab_size)
            if len(strategy_specs) == 0:
                output_schema_line = '- Output ONLY valid JSON with the schema {"candidates": ["...", "..."]}.\n'
            else:
                output_schema_line = '- Output ONLY valid JSON with the schema {"candidates": ["turn the scene into...", "add a...", ...]}.\n'
        forbidden_categories_instruction = format_flux2_attribute_constraint(strategy_specs)
        strategy_free_prompt_mode = bool(len(strategy_specs) == 0)
        if strategy_free_prompt_mode:
            intro_text = "You are a visual consistency tester. Your goal is to find 'Trojan' editing commands for a text-to-image model. These commands must be contextually plausible and benign, yet act as strong latent attractors that subtly disrupt the generated output."
            context_block = f'Current best objective score: {float(best_objective) + 10.0:.6f}.\n\n'
            task_target = 'the editable slot'
            action_verb_instruction = '- CRITICAL: ALL candidates MUST be formulated as concise direct editing instructions starting with action verbs.\n'
            candidate_style_instruction = ''
            disallowed_terms_instruction = ''
        else:
            intro_text = "You are a visual consistency tester. Your goal is to find 'Trojan' editing commands for a text-to-image model. These commands must be so contextually perfect and benign that they seamlessly camouflage into the original scene, yet act as strong latent attractors that subtly disrupt the generated output."
            context_block = f'Step: {int(step_idx)}\nPrompt template with slot marker: "{marked_prompt}"\nCurrent {slot_marker} value: "{current_value}"\n{slot_spec['anchor_label']}: "{slot_topic}"\nCurrent best objective score: {float(best_objective) + 10.0:.6f}.\n\n'
            task_target = slot_marker
            action_verb_instruction = "- CRITICAL: ALL candidates MUST be formulated as direct editing instructions starting with action verbs, such as 'turn the scene into', 'add', 'change the lighting to', or 'apply a'.\n "
            candidate_style_instruction = "- Favor candidates that request subtle, plausible modifications (e.g., 'turn the scene into a misty morning', 'add soft shadows', 'change the material to brushed metal') rather than overt or destructive visual conflicts.\n"
            disallowed_terms_instruction = "- Do NOT use the word 'overlay' and words related to 'cybernetic', 'neon' and strong 'fog'.\n"
        generation_prompt = f'{intro_text}\n\n{visual_feedback_block}{context_block}Feedback from previous steps:\n{feedback_block}\n\nTask:\n- Generate exactly {int(total_candidate_count)} unique candidate replacement instructions for {task_target}.\n- Prefer short, concrete, highly visual commands.\n{action_verb_instruction}- Do not output standalone nouns, adjectives, or simple prepositions.\n{candidate_style_instruction}{disallowed_terms_instruction}{forbidden_categories_instruction}- The instructions MUST make logical and semantic sense for the original context. They should appear as perfectly normal user requests to a human observer.\n{naturalness_feedback_instruction}{strategy_instruction}- Do NOT repeat any exact candidate already listed in the feedback block.\n{class_requirement}{slot_spec['candidate_requirement']}{slot_spec['relevance_requirement']}- Do not write explanations, numbering, markdown, or sentences.\n{output_schema_line}'
    else:
        generation_prompt = f'You are an AI red-teamer and iterative optimizer. Your goal is to maximize objective and to discover adversarial replacement words for the {slot_marker} slot in an image-edit prompt that cause the target image generation model to fail, produce unexpected artifacts, or significantly deviate from its intended behavior.\n\n{visual_feedback_block}Step: {int(step_idx)}\nPrompt template with slot marker: "{marked_prompt}"\nCurrent {slot_marker} value: "{current_value}"\n{slot_spec['anchor_label']}: "{slot_topic}"\nCurrent best objective score: {float(best_objective) + 10.0:.6f}.\n\nFeedback from previous steps:\n{feedback_block}\n\nTask:\n- Generate exactly {int(args.gcg_scene_vocab_size)} unique candidate replacements for {slot_marker}.\n{naturalness_feedback_instruction}{strategy_instruction}- Do NOT repeat any exact candidate already listed in the feedback block.\n{class_requirement}- The candidates do NOT need to make logical sense in the original context. In fact, unusual, contradictory, or visually disruptive objects/attributes often yield better adversarial results.\n- Focus on concrete, visualizable terms (objects, textures, lighting, strange combinations) rather than abstract concepts.\n{slot_spec['candidate_requirement']}{slot_spec['relevance_requirement']}- Do not write explanations, numbering, markdown, or sentences.\n- Output ONLY valid JSON with the schema {{"candidates": ["word1", "word2", ...]}}.\n'
    raw_answer, error = query_vlm_text(image_path=reference_image_path, question=generation_prompt, vlm_backend=str(args.gcg_scene_llm_backend), vlm_model_id=str(args.gcg_scene_llm_model_id), vlm_device_raw=str(args.gcg_scene_llm_device), max_new_tokens=int(args.gcg_scene_llm_max_new_tokens), enable_thinking=bool(args.gcg_scene_llm_thinking), do_sample=bool(args.gcg_scene_llm_do_sample), classifier_name=str(getattr(args, 'classifier_name', '')), runtime_cache=runtime_cache)
    if error is not None:
        return ([], str(raw_answer or ''), generation_prompt, error)
    if flux2_strategy_prompt_mode:
        strategy_groups = parse_scene_vocab_strategy_groups(str(raw_answer or ''), prompts_per_strategy=int(prompts_per_strategy), strategy_specs=strategy_specs)
        strategy_entries = flatten_scene_vocab_strategy_groups(strategy_groups)
        words = [str(item['word']) for item in strategy_entries]
        if len(words) == 0:
            fallback_candidate = normalize_slot_value(str(raw_answer or ''), fallback_word, slot_kind)
            if fallback_candidate:
                first_spec = strategy_specs[0]
                strategy_groups = [{'name': str(first_spec['name']), 'title': str(first_spec['title']), 'candidates': [fallback_candidate]}, *[{'name': str(spec['name']), 'title': str(spec['title']), 'candidates': []} for spec in strategy_specs[1:]]]
                strategy_entries = flatten_scene_vocab_strategy_groups(strategy_groups)
                words = [str(item['word']) for item in strategy_entries]
        setattr(args, '_scene_vocab_strategy_groups', list(strategy_groups))
        setattr(args, '_scene_vocab_strategy_entries', list(strategy_entries))
    else:
        parsed_limit = max(1, int(args.gcg_scene_vocab_size))
        words = parse_scene_vocab_words(str(raw_answer or ''), limit=parsed_limit)
        if len(words) == 0:
            fallback_candidate = normalize_slot_value(str(raw_answer or ''), fallback_word, slot_kind)
            if fallback_candidate:
                words = [fallback_candidate]
    return (words, str(raw_answer or ''), generation_prompt, None)

def query_vlm_text(*, image_path: Path, question: str, vlm_backend: str, vlm_model_id: str, vlm_device_raw: str, max_new_tokens: int, enable_thinking: bool, do_sample: bool, classifier_name: str='', runtime_cache: Optional[PersistentVLMRuntimeCache]=None) -> Tuple[str, Optional[str]]:
    _maybe_disable_cudnn_sdpa_for_vim_small(classifier_name, context='query_vlm_text')
    if runtime_cache is not None:
        return runtime_cache.query(image_path=image_path, question=question, vlm_backend=vlm_backend, vlm_model_id=vlm_model_id, vlm_device_raw=vlm_device_raw, max_new_tokens=max_new_tokens, enable_thinking=enable_thinking, do_sample=do_sample)
    vlm_model = None
    vlm_processor = None
    ask_fn = None
    uses_pipeline_backend = False
    device = resolve_vlm_device(vlm_device_raw)
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    try:
        backend = infer_vlm_backend(vlm_backend=str(vlm_backend), model_id=str(vlm_model_id), allow_blip=True)
        vlm_model, vlm_processor, ask_fn, uses_pipeline_backend = load_vlm_runtime(backend=backend, model_id=str(vlm_model_id), vlm_dtype=dtype, vlm_device=device, allow_blip=True)
        if not uses_pipeline_backend:
            vlm_model = vlm_model.to(device)
            vlm_model.eval()
        image = Image.open(image_path).convert('RGB')
        with torch.no_grad():
            raw_answer = ask_fn(image=image, question=str(question), model=vlm_model, processor=vlm_processor, device=device, max_new_tokens=int(max_new_tokens), enable_thinking=bool(enable_thinking), do_sample=bool(do_sample))
        return (str(raw_answer or '').strip(), None)
    except Exception as exc:
        return ('', str(exc))
    finally:
        if vlm_model is not None and hasattr(vlm_model, 'to') and (not uses_pipeline_backend):
            try:
                vlm_model.to('cpu')
            except Exception:
                pass
        gc.collect()
        if device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

def query_vlm_word(*, image_path: Path, args: argparse.Namespace, slot_kind: str, fallback_word: str, runtime_cache: Optional[PersistentVLMRuntimeCache]=None) -> Tuple[str, str, Optional[str]]:
    raw_answer, error = query_vlm_text(image_path=image_path, question=str(args.scene_vlm_question), vlm_backend=str(args.scene_vlm_backend), vlm_model_id=str(args.scene_vlm_model_id), vlm_device_raw=str(args.scene_vlm_device), max_new_tokens=int(args.scene_vlm_max_new_tokens), enable_thinking=bool(args.scene_vlm_thinking), do_sample=bool(args.scene_vlm_do_sample), classifier_name=str(getattr(args, 'classifier_name', '')), runtime_cache=runtime_cache)
    if error is not None:
        return (fallback_word, '', error)
    candidate_word = normalize_slot_value(raw_answer, fallback_word, slot_kind)
    return (candidate_word, raw_answer, None)

def image_tensor_01_to_pil(image_01: torch.Tensor) -> Image.Image:
    tensor = image_01
    if tensor.ndim == 4:
        if int(tensor.shape[0]) < 1:
            raise ValueError('image tensor batch is empty')
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f'expected CHW tensor, got shape={tuple(tensor.shape)}')
    if int(tensor.shape[0]) not in {1, 3}:
        raise ValueError(f'expected 1 or 3 channels, got shape={tuple(tensor.shape)}')
    chw = tensor.detach().to(device='cpu', dtype=torch.float32)
    if int(chw.shape[0]) == 1:
        chw = chw.repeat(3, 1, 1)
    chw = torch.clamp(chw, 0.0, 1.0)
    hwc = (chw.permute(1, 2, 0).contiguous().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(hwc, mode='RGB')

def image_to_tensor_01(image: Image.Image) -> torch.Tensor:
    arr = torch.from_numpy(np.array(image.convert('RGB'), dtype=np.float32))
    arr = arr.permute(2, 0, 1).contiguous() / 255.0
    return arr.unsqueeze(0)

def classifier_input_image(image: Image.Image, classifier) -> Image.Image:
    """Materialize the RGB image tensor actually presented to the victim classifier."""
    input_size = 224
    for attr_name in ('input_res', 'input_size'):
        try:
            value = int(getattr(classifier, attr_name, 0))
        except Exception:
            continue
        if value > 0:
            input_size = value
            break
    tensor = image_to_tensor_01(image)
    if tuple(tensor.shape[-2:]) != (input_size, input_size):
        tensor = F.interpolate(tensor, size=(input_size, input_size), mode='bilinear', align_corners=False, antialias=True)
    return image_tensor_01_to_pil(tensor).convert('RGB')

def evaluate_attack_candidates(*, args: argparse.Namespace, classifier: TorchvisionClassifier, candidate_words: Sequence[str], candidate_prompts: Sequence[str], has_input_image: bool, render_session: Optional[PersistentFluxRenderSession]=None, mixed_initial_edit_cache: Optional[Dict[str, object]]=None, capture_classifier_tile_image: bool=True, **_unused_legacy_options) -> Tuple[List[Dict[str, object]], Optional[str], None]:
    """Render and score edit candidates only; no inversion or source-image tile is rendered."""
    del _unused_legacy_options
    words = [str(item) for item in candidate_words]
    prompts = [str(item) for item in candidate_prompts]
    if len(prompts) == 0:
        return ([], 'no_candidates', None)
    if len(words) != len(prompts):
        return ([], f'candidate_count_mismatch:{len(words)}!={len(prompts)}', None)

    if render_session is None:
        raise ValueError('isolated attack runtime requires an initialized render_session')
    images = render_session.render_images(prompts=prompts, mixed_initial_edit_cache=mixed_initial_edit_cache)
    if len(images) != len(prompts):
        return ([], f'image_count_mismatch:{len(images)}!={len(prompts)}', None)
    prompt_query_count = 0
    setattr(render_session, 'last_prompt_query_count', 0)
    results: List[Dict[str, object]] = []
    for index, rendered_image in enumerate(images):
        tile = rendered_image.convert('RGB').copy()
        try:
            image_01 = image_to_tensor_01(tile).to(device=str(args.device))
            prompt_query_count += 1
            setattr(render_session, 'last_prompt_query_count', int(prompt_query_count))
            with torch.no_grad():
                objective, stats = classifier.objective_and_stats(image_01, target_label=None)
        except Exception as exc:
            return (results, f'candidate_{int(index)}:{type(exc).__name__}:{exc}', None)
        result: Dict[str, object] = {'candidate_word': words[index], 'candidate_prompt': prompts[index], 'candidate_objective': float(objective), 'pred_idx': stats.get('pred_idx'), 'pred_conf': stats.get('pred_conf'), 'pred_logit': stats.get('pred_logit'), 'target_conf': stats.get('target_conf'), 'target_logit': stats.get('target_logit'), 'target_label_conf': stats.get('target_label_conf'), 'target_label_logit': stats.get('target_label_logit'), 'ce': stats.get('ce'), 'candidate_variant': 'prompt', 'candidate_strip_index': int(index), 'candidate_selected_image': tile.copy(), 'candidate_selected_image_width': int(tile.width), 'candidate_selected_image_height': int(tile.height), 'candidate_selected_image_source': 'raw_tile'}
        results.append(result)
        if bool(capture_classifier_tile_image):
            try:
                evaluated_image = classifier_input_image(tile, classifier)
                result['candidate_classifier_image'] = evaluated_image.copy()
                result['candidate_classifier_image_size'] = int(evaluated_image.size[0])
            except Exception as exc:
                return (results, f'candidate_{int(index)}_capture:{type(exc).__name__}:{exc}', None)
    return (results, None, None)

def relpath_from_run_dir(run_dir: Path, path: Path) -> str:
    run_dir_abs = run_dir.resolve()
    path_abs = path.resolve()
    try:
        return str(path_abs.relative_to(run_dir_abs))
    except Exception:
        return str(path_abs)

def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

def save_blackbox_prompt_artifacts(*, run_dir: Path, step_idx: int, candidate_source: str, prompt_text: str, raw_answer: str, feedback_used: Sequence[Dict[str, object]], generated_words: Sequence[str], filtered_words: Sequence[str], scored_candidates: Sequence[Dict[str, object]], vlm_error: Optional[str], score_error: Optional[str]) -> Dict[str, str]:

    def _json_safe_candidate(item: Dict[str, object]) -> Dict[str, object]:
        return {key: value for key, value in dict(item).items() if key not in {'candidate_classifier_image', 'candidate_selected_image', 'candidate_precomputed_selected_image_path'}}
    artifact_dir = run_dir / 'prompt_artifacts'
    human_step = int(step_idx) + 1
    stem = 'scene_vocab' if str(candidate_source) == 'gemma_scene_vocab' else 'vlm_query'
    prompt_text_path = artifact_dir / f'step_{human_step:03d}_{stem}_prompt.txt'
    response_json_path = artifact_dir / f'step_{human_step:03d}_{stem}_response.json'
    raw_answer_text_path = artifact_dir / f'step_{human_step:03d}_{stem}_raw_answer.txt'
    raw_answer_text = str(raw_answer or '').replace('\r\n', '\n').replace('\r', '\n')
    prompt_text_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_text_path.write_text(str(prompt_text or ''), encoding='utf-8')
    raw_answer_text_path.write_text(raw_answer_text, encoding='utf-8')
    response_payload: Dict[str, object] = {'step_idx': int(step_idx), 'step': human_step, 'requested_candidate_source': str(candidate_source), 'used_candidate_source': str(candidate_source), 'fallback_to_grad': False, 'error': None, 'feedback_used': list(feedback_used), 'raw_answer': raw_answer_text, 'raw_answer_lines': raw_answer_text.split('\n'), 'raw_answer_text_path': relpath_from_run_dir(run_dir, raw_answer_text_path), 'generated_words': list(generated_words), 'filtered_words': list(filtered_words), 'selected_words': [str(item.get('candidate_word', '')) for item in scored_candidates], 'scored_candidates': [_json_safe_candidate(item) for item in scored_candidates]}
    if vlm_error is not None:
        response_payload['error'] = str(vlm_error)
    if score_error is not None:
        response_payload['score_error'] = str(score_error)
    write_json(response_json_path, response_payload)
    return {'prompt_text_path': relpath_from_run_dir(run_dir, prompt_text_path), 'response_json_path': relpath_from_run_dir(run_dir, response_json_path)}
__all__ = ('PersistentVLMRuntimeCache', 'classifier_input_image', 'evaluate_attack_candidates', 'generate_scene_vocab_words', 'image_to_tensor_01', 'query_vlm_word', 'save_blackbox_prompt_artifacts')
