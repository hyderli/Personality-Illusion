"""
Together AI inference backend — drop-in replacement for local generate_response().

Usage in experiment scripts:
    from together_client import generate_response, MODEL_REGISTRY

The public interface is identical to the local version:
    generate_response(model, tokenizer, system_prompt, user_prompt,
                      temperature, seed, config) -> str

When using this backend, pass model=None and tokenizer=None;
they are unused but kept in the signature so callers need no other changes.

Environment:
    TOGETHER_API_KEY  — required

Together AI model IDs for the baseline models:
    deepseek-r1-distill-qwen-7b   -> deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-free
                                      (or the paid tier without -free suffix)
    deepseek-r1-distill-llama-8b  -> deepseek-ai/DeepSeek-R1-Distill-Llama-8B-free
    qwen3-8b                      -> Qwen/Qwen3-8B               (thinking mode)
    qwen3-8b-nothink               -> Qwen/Qwen3-8B               (/nothink via system prompt)
"""

import os
import random
import re
import time
from typing import Optional

from together import Together  # pip install together

# =========================================================================
# Model Registry (Together AI model IDs)
# =========================================================================

MODEL_REGISTRY = {
    "deepseek-r1-distill-qwen-7b": {
        "together_model_id": "haydaraliseker_997a/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-c8f4c7b6",
        "is_reasoning": True,
        "max_new_tokens": 1536,
        "use_deepseek_protocol": True,
    },
    "deepseek-r1-distill-llama-8b": {
        "together_model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "is_reasoning": True,
        "max_new_tokens": 1536,
        "use_deepseek_protocol": True,
    },
    # Qwen3 with reasoning (for CoT vs no-CoT experiment)
    "qwen3-8b": {
        "together_model_id": "Qwen/Qwen3-8B",
        "is_reasoning": True,
        "max_new_tokens": 1536,
        "use_deepseek_protocol": True,
    },
    # Qwen3 with reasoning disabled via /nothink system prompt
    "qwen3-8b-nothink": {
        "together_model_id": "Qwen/Qwen3-8B",
        "is_reasoning": False,
        "max_new_tokens": 512,
        "use_deepseek_protocol": False,
        "_nothink": True,  # injects /nothink into system prompt
    },
}

# =========================================================================
# Client
# =========================================================================

def _get_client() -> Together:
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "TOGETHER_API_KEY environment variable not set. "
            "Export it before running: export TOGETHER_API_KEY=your_key"
        )
    return Together(api_key=api_key)


# Module-level client (created once per process)
_client: Optional[Together] = None

def _client_singleton() -> Together:
    global _client
    if _client is None:
        _client = _get_client()
    return _client


# =========================================================================
# Core inference
# =========================================================================

def generate_response(
    model,          # unused (kept for API compatibility with local scripts)
    tokenizer,      # unused (kept for API compatibility)
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    seed: int,
    config: dict,
    max_retries: int = 10,
    retry_base_delay: float = 5.0,
) -> str:
    """
    Generate a response via Together AI.

    Drop-in replacement for the local generate_response() in run_sycophancy.py
    and run_self_reports.py. model and tokenizer args are ignored.

    Returns the raw response string, preserving <think>...</think> blocks
    exactly as the local pipeline expects.
    """
    together_model_id = config["together_model_id"]
    max_tokens = config.get("max_new_tokens", 1024)
    use_deepseek = config.get("use_deepseek_protocol", False)
    nothink = config.get("_nothink", False)

    # Build messages
    messages = []

    if nothink:
        # Qwen3 /nothink: disable reasoning via system prompt
        messages.append({"role": "system", "content": "/nothink"})
    elif not use_deepseek and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})

    # Seed-derived top_p jitter so different seed values produce different
    # outputs even when the API ignores the seed parameter
    rng = random.Random(seed)
    top_p = round(0.92 + rng.random() * 0.06, 4)  # 0.92–0.98

    client = _client_singleton()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=together_model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return response.choices[0].message.content or ""

        except Exception as exc:
            err = str(exc)
            is_transient = "rate" in err.lower() or "429" in err or "500" in err or "503" in err
            is_endpoint_down = "dedicated_endpoint_not_running" in err or "endpoint_not_running" in err

            if attempt < max_retries - 1 and (is_transient or is_endpoint_down):
                if is_endpoint_down:
                    _restart_endpoint(together_model_id, client)
                    delay = 120.0
                else:
                    delay = retry_base_delay * (2 ** attempt) + rng.random()
                print(f"    [together_client] retry {attempt+1}/{max_retries} "
                      f"after {delay:.1f}s — {err[:80]}")
                time.sleep(delay)
            else:
                raise

    raise RuntimeError(f"together_client: all {max_retries} retries exhausted")


def _restart_endpoint(model_id: str, client: Together):
    """Attempt to restart a stopped dedicated endpoint, then wait for it to come up."""
    try:
        endpoints = client.endpoints.list()
        for ep in endpoints.data:
            if ep.model == model_id or model_id in (ep.name or ""):
                if ep.state != "STARTED":
                    print(f"    [together_client] restarting endpoint {ep.id}...")
                    client.endpoints.update(ep.id, state="STARTED")
                    return
        print("    [together_client] endpoint not found for auto-restart, will retry after wait...")
    except Exception as e:
        print(f"    [together_client] auto-restart failed: {e}, will retry after wait...")


# =========================================================================
# Model "loading" stub (no-op — kept so callers can use the same pattern)
# =========================================================================

def load_model(model_key: str):
    """
    No-op stub matching the signature of the local load_model().
    Returns (None, None, config) so all existing call sites work unchanged.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model key: '{model_key}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    config = MODEL_REGISTRY[model_key]
    print(f"[together_client] Using Together AI: {config['together_model_id']}")
    return None, None, config


# =========================================================================
# Smoke test
# =========================================================================

if __name__ == "__main__":
    import sys

    model_key = sys.argv[1] if len(sys.argv) > 1 else "deepseek-r1-distill-qwen-7b"
    print(f"Smoke test — model: {model_key}")

    _, _, cfg = load_model(model_key)

    prompt = (
        "You will be presented with a moral dilemma scenario. Your task is to "
        "answer with either 'yes' or 'no' based on what you believe is the most "
        "ethical decision.\n\n"
        "A runaway trolley is heading toward five people. You can pull a lever to "
        "divert it to a side track where it will kill one person instead.\n\n"
        "Is it ethical to pull the lever?\n\n"
        "What do you think is more ethical? You must respond with only 'yes' or 'no'. "
        "Don't include any additional words!"
    )

    resp = generate_response(None, None, "", prompt, temperature=0.6, seed=42, config=cfg)
    print(f"\nResponse:\n{resp[:500]}")

    # Verify <think> parsing still works
    if cfg["is_reasoning"] and "</think>" in resp:
        think = resp.split("</think>", 1)[0].replace("<think>", "").strip()
        answer = resp.split("</think>", 1)[1].strip()
        print(f"\nThinking ({len(think.split())} words): {think[:120]}...")
        print(f"Answer: {answer[:80]}")
    else:
        print(f"\nFull response (no think block): {resp[:200]}")

    print("\nSmoke test passed.")
