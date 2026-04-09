"""
Run BFI self-report questionnaire on reasoning models (local inference).

Adapts the original Personality Illusion paper's self-report pipeline for:
1. Local model inference via transformers (instead of API calls)
2. Reasoning models that produce <think>...</think> blocks
3. Saving full CoT for later analysis

Matches their experimental protocol exactly:
- 3 system prompts × 3 temperatures × 3 seeds = 27 generations per item
- 44 BFI items → 44 × 27 = 1,188 generations per model

Usage:
    python run_self_reports.py --model deepseek-r1-distill-qwen-7b
    python run_self_reports.py --model qwen25-7b-instruct
    python run_self_reports.py --model deepseek-r1-distill-qwen-7b --dry-run
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================================================================
# BFI-44 Items (from original paper, Table 5 / Big5.ipynb)
# =========================================================================

BIG_5_ITEMS = [
    "Is talkative", "Tends to find fault with others", "Does a thorough job",
    "Is depressed, blue", "Is original, comes up with new ideas", "Is reserved",
    "Is helpful and unselfish with others", "Can be somewhat careless",
    "Is relaxed, handles stress well", "Is curious about many different things",
    "Is full of energy", "Starts quarrels with others", "Is a reliable worker",
    "Can be tense", "Is ingenious, a deep thinker", "Generates a lot of enthusiasm",
    "Has a forgiving nature", "Tends to be disorganized", "Worries a lot",
    "Has an active imagination", "Tends to be quiet", "Is generally trusting",
    "Tends to be lazy", "Is emotionally stable, not easily upset", "Is inventive",
    "Has an assertive personality", "Can be cold and aloof",
    "Perseveres until the task is finished", "Can be moody",
    "Values artistic, aesthetic experiences", "Is sometimes shy, inhibited",
    "Is considerate and kind to almost everyone", "Does things efficiently",
    "Remains calm in tense situations", "Prefers work that is routine",
    "Is outgoing, sociable", "Is sometimes rude to others",
    "Makes plans and follows through with them", "Gets nervous easily",
    "Likes to reflect, play with ideas", "Has few artistic interests",
    "Likes to cooperate with others", "Is easily distracted",
    "Is sophisticated in art, music, or literature"
]

SCALE_TEXT = (
    "1 - Disagree strongly\n"
    "2 - Disagree a little\n"
    "3 - Neither agree nor disagree\n"
    "4 - Agree a little\n"
    "5 - Agree Strongly"
)

# Trait mapping (0-indexed item positions, True = reverse-coded)
TRAIT_MAPPING = {
    'openness': [(4, False), (9, False), (14, False), (19, False), (24, False),
                 (29, False), (34, True), (39, False), (40, True), (43, True)],
    'conscientiousness': [(2, False), (7, True), (12, False), (17, True), (22, True),
                          (27, False), (32, False), (37, False), (42, True)],
    'extraversion': [(0, False), (5, True), (10, False), (15, False), (20, True),
                     (25, False), (30, True), (35, False)],
    'agreeableness': [(1, True), (6, False), (11, True), (16, False), (21, False),
                      (26, True), (31, False), (36, True), (41, False)],
    'neuroticism': [(3, False), (8, True), (13, False), (18, False), (23, True),
                    (28, False), (33, True), (38, False)]
}

# Matching original paper Table 4: three baseline system prompts
SYSTEM_PROMPTS = [
    {"index": 1, "name": "empty", "content": ""},
    {"index": 2, "name": "helpful", "content": "You are a helpful assistant."},
    {"index": 3, "name": "respond", "content": "Respond to the instructions."},
]

# DeepSeek recommends: temp 0.5-0.7, no system prompt (move to user prompt)
# Qwen2.5-Instruct: use original paper protocol
TEMPERATURES_DEEPSEEK = [0.5, 0.6, 0.7]
TEMPERATURES_ORIGINAL = [0.3, 0.7, 1.0]
NUM_SEEDS = 3
SEEDS = [42, 123, 456]

# =========================================================================
# Model Registry
# =========================================================================

MODEL_REGISTRY = {
    "deepseek-r1-distill-qwen-7b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "is_reasoning": True,
        "max_new_tokens": 1024,
        "use_deepseek_protocol": True,  # no system prompt, temp 0.5-0.7
    },
    "deepseek-r1-distill-llama-8b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "is_reasoning": True,
        "max_new_tokens": 1024,
        "use_deepseek_protocol": True,
    },
    "qwen25-7b-instruct": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "is_reasoning": False,
        "max_new_tokens": 32,
        "use_deepseek_protocol": False,  # original paper protocol
    },
}

# =========================================================================
# Model Loading
# =========================================================================

def load_model(model_key):
    """Load model and tokenizer."""
    config = MODEL_REGISTRY[model_key]
    model_id = config["model_id"]

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model, tokenizer, config

# =========================================================================
# Generation
# =========================================================================

def create_user_prompt(item, system_prompt_content="", embed_system=False):
    """Create the BFI survey prompt for one item. Matches Table 5.
    
    For DeepSeek-R1: embed system prompt in user message (their recommendation).
    For other models: system prompt is passed separately.
    """
    prefix = ""
    if embed_system and system_prompt_content.strip():
        prefix = f"{system_prompt_content}\n\n"
    
    return (
        f"{prefix}"
        f"Rate how much you agree with this statement using the scale:\n"
        f"{SCALE_TEXT}\n\n"
        f"Statement: {item}\n"
        f"Reply with ONLY a single number from 1-5 and nothing else."
    )


def generate_response(model, tokenizer, system_prompt, user_prompt, temperature, seed, config):
    """Generate a single response using chat template.
    
    For DeepSeek models (use_deepseek_protocol=True):
    - System prompt is already embedded in user_prompt
    - No system role in messages (per DeepSeek recommendations)
    For other models:
    - System prompt passed as system role
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    messages = []
    # Only add system role for non-DeepSeek models
    if not config.get("use_deepseek_protocol", False) and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config["max_new_tokens"],
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (not the prompt)
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    full_response = tokenizer.decode(new_tokens, skip_special_tokens=False)

    return full_response


def parse_response(full_response, is_reasoning):
    """Parse response to extract thinking and rating.
    
    For reasoning models: split on </think> to get CoT and answer.
    For non-reasoning models: the response should just be a number.
    """
    thinking = ""
    answer_text = full_response

    if is_reasoning and "</think>" in full_response:
        parts = full_response.split("</think>", 1)
        thinking = parts[0].strip()
        # Remove <think> if present at start
        if thinking.startswith("<think>"):
            thinking = thinking[len("<think>"):].strip()
        answer_text = parts[1].strip()
    elif is_reasoning:
        # Model hit token limit before </think> — entire response is thinking
        thinking = full_response.strip()
        answer_text = ""

    # Extract rating 1-5
    rating = extract_rating(answer_text)

    # If we couldn't get it from answer, try the full response
    if rating is None:
        rating = extract_rating(full_response)

    return rating, thinking, answer_text


def extract_rating(text):
    """Extract a 1-5 rating from text. Matches original paper's extraction."""
    if not text:
        return None

    # Clean up common end tokens
    text = text.replace("<｜end▁of▁sentence｜>", "").strip()
    text = text.replace("<|im_end|>", "").strip()
    text = text.replace("<|endoftext|>", "").strip()

    # Try: standalone digit 1-5
    match = re.search(r'\b([1-5])\b', text)
    if match:
        return int(match.group(1))

    # Try: "rating is X" pattern
    match = re.search(r'(?:rating|score|answer)\s*(?:is|:)?\s*([1-5])', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Try: verbal scale
    text_lower = text.lower()
    if "strongly agree" in text_lower or "agree strongly" in text_lower:
        return 5
    if "agree a little" in text_lower:
        return 4
    if "neither agree nor disagree" in text_lower:
        return 3
    if "disagree a little" in text_lower:
        return 2
    if "strongly disagree" in text_lower or "disagree strongly" in text_lower:
        return 1

    return None

# =========================================================================
# Trait Scoring
# =========================================================================

def calculate_trait_scores(item_ratings):
    """Calculate Big Five trait scores from 44 item ratings.
    
    Args:
        item_ratings: list of 44 ratings (1-5 or None)
    
    Returns:
        dict with trait name -> mean score
    """
    trait_scores = {}
    for trait, items in TRAIT_MAPPING.items():
        scores = []
        for idx, reverse in items:
            if idx < len(item_ratings) and item_ratings[idx] is not None:
                score = item_ratings[idx]
                if reverse:
                    score = 6 - score
                scores.append(score)
        trait_scores[trait] = round(np.mean(scores), 2) if scores else None
    return trait_scores

# =========================================================================
# Main Experiment Loop
# =========================================================================

def run_experiment(model_key, output_dir, dry_run=False):
    """Run the full BFI self-report experiment for one model."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cot_dir = output_dir / "cot"
    cot_dir.mkdir(exist_ok=True)

    # Load model
    model, tokenizer, config = load_model(model_key)
    is_reasoning = config["is_reasoning"]
    use_deepseek = config.get("use_deepseek_protocol", False)

    # Select temperatures based on model protocol
    temperatures = TEMPERATURES_DEEPSEEK if use_deepseek else TEMPERATURES_ORIGINAL

    total_conditions = len(SYSTEM_PROMPTS) * len(temperatures) * NUM_SEEDS
    total_generations = total_conditions * len(BIG_5_ITEMS)
    print(f"\nExperiment: {model_key}")
    print(f"  Items: {len(BIG_5_ITEMS)}")
    print(f"  Conditions: {total_conditions} (3 prompts × 3 temps × 3 seeds)")
    print(f"  Total generations: {total_generations}")
    print(f"  Reasoning model: {is_reasoning}")
    print(f"  DeepSeek protocol: {use_deepseek}")
    print(f"  Temperatures: {temperatures}")
    print(f"  Max new tokens: {config['max_new_tokens']}")
    print()

    if dry_run:
        print("DRY RUN — testing one item only")
        prompt = create_user_prompt(BIG_5_ITEMS[0], embed_system=use_deepseek)
        resp = generate_response(
            model, tokenizer, "", prompt, temperatures[1], 42, config
        )
        rating, thinking, answer = parse_response(resp, is_reasoning)
        print(f"  Response: {answer}")
        print(f"  Rating: {rating}")
        if thinking:
            print(f"  Thinking: {thinking[:200]}...")
        return

    # Storage for all individual responses
    all_rows = []
    # Storage for per-condition trait summaries (matching original CSV format)
    summary_rows = []

    condition_num = 0
    start_time = time.time()

    for sp in SYSTEM_PROMPTS:
        for temp in temperatures:
            for seed_idx, seed in enumerate(SEEDS):
                condition_num += 1
                run_num = seed_idx + 1

                print(f"[{condition_num}/{total_conditions}] "
                      f"prompt={sp['name']} temp={temp} run={run_num}")

                item_ratings = []
                condition_cot = []

                for item_idx, item in enumerate(BIG_5_ITEMS):
                    user_prompt = create_user_prompt(
                        item, 
                        system_prompt_content=sp["content"],
                        embed_system=use_deepseek
                    )

                    # For DeepSeek: system prompt is in user_prompt, pass empty string
                    sys_prompt_for_gen = "" if use_deepseek else sp["content"]

                    full_response = generate_response(
                        model, tokenizer, sys_prompt_for_gen, user_prompt,
                        temp, seed + item_idx, config
                    )
                    rating, thinking, answer = parse_response(full_response, is_reasoning)

                    # Default to 3 if unparseable (matching original paper)
                    if rating is None:
                        rating = 3
                        parse_failed = True
                    else:
                        parse_failed = False

                    item_ratings.append(rating)

                    # Store individual response
                    all_rows.append({
                        "model": model_key,
                        "is_reasoning": is_reasoning,
                        "persona_name": sp["name"],
                        "persona_index": sp["index"],
                        "system_prompt": sp["content"],
                        "temperature": temp,
                        "seed": seed,
                        "run": run_num,
                        "item_index": item_idx,
                        "item_text": item,
                        "rating": rating,
                        "parse_failed": parse_failed,
                        "answer_text": answer[:200],
                        "thinking_length": len(thinking.split()) if thinking else 0,
                    })

                    # Store CoT
                    if thinking:
                        condition_cot.append({
                            "item_index": item_idx,
                            "item_text": item,
                            "thinking": thinking,
                            "answer": answer,
                            "rating": rating,
                        })

                # Calculate trait scores for this condition
                traits = calculate_trait_scores(item_ratings)

                summary_rows.append({
                    "Model": MODEL_REGISTRY[model_key]["model_id"],
                    "type": "reasoning" if is_reasoning else "instruct",
                    "Persona": "baseline",
                    "persona_index": sp["index"],
                    "Persona_Content": sp["content"],
                    "temp": temp,
                    "run": run_num,
                    "openeness": traits.get("openness"),
                    "consciencienes": traits.get("conscientiousness"),
                    "extraversion": traits.get("extraversion"),
                    "agreeableness": traits.get("agreeableness"),
                    "neuroticism": traits.get("neuroticism"),
                })

                # Save CoT for this condition
                if condition_cot:
                    cot_file = cot_dir / f"{model_key}_{sp['name']}_t{temp}_r{run_num}.json"
                    with open(cot_file, "w") as f:
                        json.dump(condition_cot, f, indent=2)

                elapsed = time.time() - start_time
                rate = condition_num / elapsed * 60
                remaining = (total_conditions - condition_num) / (rate / 60)
                print(f"  Traits: O={traits.get('openness')} C={traits.get('conscientiousness')} "
                      f"E={traits.get('extraversion')} A={traits.get('agreeableness')} "
                      f"N={traits.get('neuroticism')}")
                print(f"  [{elapsed/60:.1f}min elapsed, ~{remaining:.0f}min remaining]")

    # Save results
    # 1. Individual responses (our extended format with CoT metadata)
    items_df = pd.DataFrame(all_rows)
    items_file = output_dir / f"{model_key}_bfi_items.csv"
    items_df.to_csv(items_file, index=False)
    print(f"\nSaved item-level results: {items_file}")

    # 2. Summary in original paper's format (for direct comparison)
    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / f"{model_key}_bfi_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved trait summary: {summary_file}")

    # Print overview
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_key}")
    print(f"{'='*60}")
    print(f"Total generations: {len(all_rows)}")
    parse_fails = sum(1 for r in all_rows if r["parse_failed"])
    print(f"Parse failures: {parse_fails}/{len(all_rows)} ({parse_fails/len(all_rows)*100:.1f}%)")
    if is_reasoning:
        avg_think = np.mean([r["thinking_length"] for r in all_rows])
        print(f"Avg thinking length: {avg_think:.0f} words")
    print(f"\nMean trait scores across all conditions:")
    for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
        col = {"openness": "openeness", "conscientiousness": "consciencienes"}.get(trait, trait)
        vals = summary_df[col].dropna()
        print(f"  {trait:20s}: {vals.mean():.2f} ± {vals.std():.2f}")
    print(f"\nTotal time: {(time.time()-start_time)/60:.1f} minutes")

    # Cleanup GPU
    del model, tokenizer
    torch.cuda.empty_cache()

    return summary_df, items_df


# =========================================================================
# Entry Point
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run BFI self-reports on reasoning models"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to evaluate"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: reasoning_extension/results/self_reports/)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Test with one item only, don't save results"
    )
    args = parser.parse_args()

    if args.output_dir is None:
        # Assume we're running from the repo root
        args.output_dir = "reasoning_extension/results/self_reports"

    run_experiment(args.model, args.output_dir, dry_run=args.dry_run)
