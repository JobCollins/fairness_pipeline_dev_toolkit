"""Tests for LLM response cache hit/miss and key composition."""

from __future__ import annotations

from fairness_pipeline_dev_toolkit.llm_evals.cache import ResponseCache, make_cache_key


def test_cache_miss_then_hit(tmp_path):
    cache = ResponseCache(cache_dir=tmp_path)
    key = make_cache_key("openai", "gpt-4o-mini", "hello", {"temperature": 0.0})

    assert cache.get(key) is None
    assert cache.misses == 1

    cache.set(key, "world")
    assert cache.get(key) == "world"
    assert cache.hits == 1
    assert cache.contains(key)


def test_cache_key_stable_for_same_inputs():
    key_a = make_cache_key("openai", "gpt-4o-mini", "prompt", {"temperature": 0.0})
    key_b = make_cache_key("openai", "gpt-4o-mini", "prompt", {"temperature": 0.0})
    assert key_a == key_b


def test_cache_key_changes_with_provider_model_prompt_or_params():
    base = make_cache_key("openai", "gpt-4o-mini", "prompt", {"temperature": 0.0})
    assert base != make_cache_key("anthropic", "gpt-4o-mini", "prompt", {"temperature": 0.0})
    assert base != make_cache_key("openai", "gpt-4o", "prompt", {"temperature": 0.0})
    assert base != make_cache_key("openai", "gpt-4o-mini", "other", {"temperature": 0.0})
    assert base != make_cache_key("openai", "gpt-4o-mini", "prompt", {"temperature": 0.5})


def test_get_or_compute_key_matches_make_cache_key():
    cache = ResponseCache(cache_dir="/tmp/unused")
    key = cache.get_or_compute_key("local", "stub", "p", {"k": 1})
    assert key == make_cache_key("local", "stub", "p", {"k": 1})
