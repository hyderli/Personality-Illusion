"""
Run BFI self-report questionnaire via Together AI (API inference).

Identical experiment logic as run_self_reports.py but uses Together AI instead
of local GPU. Output format is fully compatible with the original script.

Usage:
    export TOGETHER_API_KEY=your_key
    python run_self_reports_api.py --model deepseek-r1-distill-qwen-7b
    python run_self_reports_api.py --model deepseek-r1-distill-qwen-7b --dry-run
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from together_client import MODEL_REGISTRY, generate_response, load_model

# =========================================================================
# BFI-44 (from run_self_reports.py — unchanged)
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
    "Is sophisticated in art, music, or literature",
]

SCALE_TEXT = (
    "1 - Disagree strongly\n"
    "2 - Disagree a little\n"
    "3 - Neither agree nor disagree\n"
    "4 - Agree a little\n"
    "5 - Agree Strongly"
)

TRAIT_MAPPING = {
    "openness":          [(4,False),(9,False),(14,False),(19,False),(24,False),
                          (29,False),(34,True),(39,False),(40,True),(43,True)],
    "conscientiousness": [(2,False),(7,True),(12,False),(17,True),(22,True),
                          (27,False),(32,False),(37,False),(42,True)],
    "extraversion":      [(0,False),(5,True),(10,False),(15,False),(20,True),
                          (25,False),(30,True),(35,False)],
    "agreeableness":     [(1,True),(6,False),(11,True),(16,False),(21,False),
                          (26,True),(31,False),(36,True),(41,False)],
    "neuroticism":       [(3,False),(8,True),(13,False),(18,False),(23,True),
                          (28,False),(33,True),(38,False)],
}

SYSTEM_PROMPTS = [
    {"index": 1, "name": "empty",   "content": ""},
    {"index": 2, "name": "helpful", "content": "You are a helpful assistant."},
    {"index": 3, "name": "respond", "content": "Respond to the instructions."},
]

TEMPERATURES_DEEPSEEK = [0.5, 0.6, 0.7]
TEMPERATURES_ORIGINAL = [0.3, 0.7, 1.0]
SEEDS = [42, 123, 456]

# =========================================================================
# Prompt / Parsing (mirrors run_self_reports.py)
# =========================================================================

def create_user_prompt(item, system_prompt_content="", embed_system=False):
    prefix = f"{system_prompt_content}\n\n" if embed_system and system_prompt_content.strip() else ""
    return (
        f"{prefix}"
        f"Rate how much you agree with this statement using the scale:\n"
        f"{SCALE_TEXT}\n\n"
        f"Statement: {item}\n"
        f"Reply with ONLY a single number from 1-5 and nothing else."
    )


def extract_rating(text):
    if not text:
        return None
    for tok in ["<｜end▁of▁sentence｜>", "<|im_end|>", "<|endoftext|>"]:
        text = text.replace(tok, "")
    text = text.strip()
    m = re.search(r'\b([1-5])\b', text)
    if m:
        return int(m.group(1))
    m = re.search(r'(?:rating|score|answer)\s*(?:is|:)?\s*([1-5])', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    tl = text.lower()
    if "strongly agree" in tl or "agree strongly" in tl:
        return 5
    if "agree a little" in tl:
        return 4
    if "neither agree nor disagree" in tl:
        return 3
    if "disagree a little" in tl:
        return 2
    if "strongly disagree" in tl or "disagree strongly" in tl:
        return 1
    return None


def parse_response(full_response, is_reasoning):
    thinking, answer_text = "", full_response
    if is_reasoning and "</think>" in full_response:
        parts = full_response.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        answer_text = parts[1].strip()
    elif is_reasoning:
        thinking = full_response.strip()
        answer_text = ""
    rating = extract_rating(answer_text) or extract_rating(full_response)
    return rating, thinking, answer_text


def calculate_trait_scores(item_ratings):
    """Return dict of Big Five trait means from list of 44 ratings (1-indexed values)."""
    scores = {}
    for trait, items in TRAIT_MAPPING.items():
        vals = []
        for idx, reverse in items:
            r = item_ratings[idx]
            if r is not None:
                vals.append(6 - r if reverse else r)
        scores[trait] = round(np.mean(vals), 3) if vals else None
    return scores

# =========================================================================
# Experiment
# =========================================================================

def run_experiment(model_key, output_dir, dry_run=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, config = load_model(model_key)
    is_reasoning = config["is_reasoning"]
    use_deepseek = config.get("use_deepseek_protocol", False)
    temperatures = TEMPERATURES_DEEPSEEK if use_deepseek else TEMPERATURES_ORIGINAL

    total_conditions = len(SYSTEM_PROMPTS) * len(temperatures) * len(SEEDS)
    print(f"\nExperiment: BFI Self-Reports — {model_key}")
    print(f"  Items:      {len(BIG_5_ITEMS)}")
    print(f"  Conditions: {total_conditions}")
    print(f"  Total gens: {total_conditions * len(BIG_5_ITEMS)}")
    print()

    if dry_run:
        item = BIG_5_ITEMS[0]
        prompt = create_user_prompt(item, embed_system=use_deepseek)
        resp = generate_response(None, None, "", prompt, temperatures[1], 42, config)
        rating, think, ans = parse_response(resp, is_reasoning)
        print(f"Item: {item}")
        print(f"Rating: {rating}")
        if think:
            print(f"Thinking ({len(think.split())} words): {think[:150]}...")
        return

    all_rows = []
    condition_num = 0
    start_time = time.time()

    for sp in SYSTEM_PROMPTS:
        for temp in temperatures:
            for seed_idx, seed in enumerate(SEEDS):
                condition_num += 1
                run_num = seed_idx + 1
                sys_prompt = "" if use_deepseek else sp["content"]

                print(f"[{condition_num}/{total_conditions}] "
                      f"prompt={sp['name']} temp={temp} run={run_num}")

                ratings = []
                for item_idx, item in enumerate(BIG_5_ITEMS):
                    prompt = create_user_prompt(
                        item, sp["content"], embed_system=use_deepseek
                    )
                    resp = generate_response(
                        None, None, sys_prompt, prompt,
                        temp, seed + item_idx, config
                    )
                    rating, think, ans = parse_response(resp, is_reasoning)
                    ratings.append(rating)

                    all_rows.append({
                        "model":            model_key,
                        "is_reasoning":     is_reasoning,
                        "persona_name":     sp["name"],
                        "persona_index":    sp["index"],
                        "system_prompt":    sp["content"],
                        "temperature":      temp,
                        "seed":             seed,
                        "run":              run_num,
                        "item_index":       item_idx,
                        "item_text":        item,
                        "rating":           rating,
                        "parse_failed":     rating is None,
                        "thinking_length":  len(think.split()) if think else 0,
                    })

                trait_scores = calculate_trait_scores(ratings)
                fails = sum(1 for r in ratings if r is None)
                print(f"  parse_fails={fails}/44  "
                      + "  ".join(f"{t[:3]}={v:.2f}" for t, v in trait_scores.items() if v))

    items_df = pd.DataFrame(all_rows)
    out_file = output_dir / f"{model_key}_bfi_items.csv"
    items_df.to_csv(out_file, index=False)
    print(f"\nSaved: {out_file}")
    print(f"Total time: {(time.time()-start_time)/60:.1f} minutes")
    return items_df


# =========================================================================
# Entry Point
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BFI self-reports via Together AI")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--output-dir", default="reasoning_extension/results/behavioral_tasks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_experiment(args.model, args.output_dir, dry_run=args.dry_run)
