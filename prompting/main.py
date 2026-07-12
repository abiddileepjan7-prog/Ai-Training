import sys
import json
import argparse
import time

try:
    from transformers import pipeline, set_seed
except ImportError:
    print("Missing dependency. Run: pip install torch transformers")
    sys.exit(1)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt2"          # try "gpt2-medium" or "gpt2-large" for a stronger (still base) model
MAX_NEW_TOKENS = 80             # keep small -- GPT-2 rambles/repeats past this
SEED = 42                       # fixed seed so runs are reproducible for comparison

generator = None  # lazily initialized in main()


def call_model(prompt: str) -> str:
    """Generate text continuation from the local GPT-2 model."""
    set_seed(SEED)
    output = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    full_text = output[0]["generated_text"]
    # Return only the newly generated continuation, not the echoed prompt
    return full_text[len(prompt):].strip()


# ---------------------------------------------------------------------------
# TASK 1: SENTIMENT CLASSIFICATION
# ---------------------------------------------------------------------------

SENTIMENT_INPUT = "The battery life is amazing but the camera is mediocre and the price feels too high."

def sentiment_prompts():
    zero_shot = f"""Classify the sentiment of this review as exactly one word: Positive, Negative, or Neutral.

Review: "{SENTIMENT_INPUT}"
Sentiment:"""

    one_shot = f"""Review: "This phone is fantastic, best purchase ever!"
Sentiment: Positive

Review: "{SENTIMENT_INPUT}"
Sentiment:"""

    few_shot = f"""Review: "Terrible, broke in two days."
Sentiment: Negative

Review: "Works fine, nothing special."
Sentiment: Neutral

Review: "Absolutely love it, exceeded expectations!"
Sentiment: Positive

Review: "Mixed feelings, some parts great, others disappointing."
Sentiment: Neutral

Review: "{SENTIMENT_INPUT}"
Sentiment:"""

    cot = f"""Review: "{SENTIMENT_INPUT}"
Let's think step by step about the good and bad points, then decide.
Reasoning:"""
    return {"zero_shot": zero_shot, "one_shot": one_shot, "few_shot": few_shot, "cot": cot}


# ---------------------------------------------------------------------------
# TASK 2: SIMPLE MATH WORD PROBLEM (kept simple -- GPT-2 base struggles with big math regardless)
# ---------------------------------------------------------------------------

MATH_INPUT = "A bakery has 20 loaves. They sell 8, then bake 5 more. How many loaves now?"

def math_prompts():
    zero_shot = f"""Solve this problem and give only the final numeric answer.

{MATH_INPUT}
Answer:"""

    one_shot = f"""Problem: A shop has 10 apples, sells 3, then gets 2 more delivered. How many apples now?
Answer: 9

Problem: {MATH_INPUT}
Answer:"""

    few_shot = f"""Problem: A shop has 10 apples, sells 3, then gets 2 more delivered. How many apples now?
Answer: 9

Problem: A warehouse has 50 boxes, ships out 10, then receives 5 more. How many boxes now?
Answer: 45

Problem: A library has 30 books, lends out 8, gets 2 returned. How many books on shelf?
Answer: 24

Problem: {MATH_INPUT}
Answer:"""

    cot = f"""{MATH_INPUT}
Let's think step by step.
Step 1:"""
    return {"zero_shot": zero_shot, "one_shot": one_shot, "few_shot": few_shot, "cot": cot}


# ---------------------------------------------------------------------------
# TASK 3: STRUCTURED EXTRACTION TO JSON
# ---------------------------------------------------------------------------

EXTRACTION_INPUT = "Contact us: Priya Nair, Senior Engineer at Solaris Labs, reachable at priya.nair@solaris.io."

def extraction_prompts():
    zero_shot = f"""Extract the contact details from this text as JSON with keys: name, title, company, email.

Text: "{EXTRACTION_INPUT}"
JSON:"""

    one_shot = f"""Text: "Reach John Doe, CTO of Nimbus Inc, at john@nimbus.com."
JSON: {{"name": "John Doe", "title": "CTO", "company": "Nimbus Inc", "email": "john@nimbus.com"}}

Text: "{EXTRACTION_INPUT}"
JSON:"""

    few_shot = f"""Text: "Reach John Doe, CTO of Nimbus Inc, at john@nimbus.com."
JSON: {{"name": "John Doe", "title": "CTO", "company": "Nimbus Inc", "email": "john@nimbus.com"}}

Text: "Contact Maria Gomez, Marketing Lead at Blue Ocean Co, maria.g@blueocean.com."
JSON: {{"name": "Maria Gomez", "title": "Marketing Lead", "company": "Blue Ocean Co", "email": "maria.g@blueocean.com"}}

Text: "{EXTRACTION_INPUT}"
JSON:"""

    cot = f"""Text: "{EXTRACTION_INPUT}"
First find the name, then the title, then the company, then the email. Then write the JSON.
Name found:"""
    return {"zero_shot": zero_shot, "one_shot": one_shot, "few_shot": few_shot, "cot": cot}


# ---------------------------------------------------------------------------
# TASK 4: SIMPLE LOGIC / COMMON SENSE QUESTION
# ---------------------------------------------------------------------------

RIDDLE_INPUT = "If all cats are animals, and Whiskers is a cat, is Whiskers an animal? Answer yes or no and explain briefly."

def riddle_prompts():
    zero_shot = f"""{RIDDLE_INPUT}
Answer:"""

    one_shot = f"""Question: If all birds can fly, and Tweety is a bird, can Tweety fly? Answer yes or no and explain briefly.
Answer: Yes, because Tweety is a bird and all birds can fly.

Question: {RIDDLE_INPUT}
Answer:"""

    few_shot = one_shot  # only one good analogous example fits GPT-2's short context well;
    # note in the guide: this itself demonstrates that few-shot isn't always better than one-shot

    cot = f"""{RIDDLE_INPUT}
Let's think step by step.
First,"""
    return {"zero_shot": zero_shot, "one_shot": one_shot, "few_shot": few_shot, "cot": cot}


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------

TASKS = {
    "sentiment": ("Sentiment Classification", sentiment_prompts),
    "math": ("Simple Math Word Problem", math_prompts),
    "extraction": ("Structured JSON Extraction", extraction_prompts),
    "riddle": ("Simple Logic Question", riddle_prompts),
}

TECHNIQUE_ORDER = ["zero_shot", "one_shot", "few_shot", "cot"]
TECHNIQUE_LABELS = {
    "zero_shot": "ZERO-SHOT",
    "one_shot": "ONE-SHOT",
    "few_shot": "FEW-SHOT",
    "cot": "CHAIN-OF-THOUGHT",
}


def run_task(task_key: str, results_log: dict):
    label, prompt_fn = TASKS[task_key]
    prompts = prompt_fn()

    print("\n" + "=" * 80)
    print(f"TASK: {label}")
    print("=" * 80)

    results_log[task_key] = {}

    for technique in TECHNIQUE_ORDER:
        prompt = prompts[technique]
        print(f"\n--- {TECHNIQUE_LABELS[technique]} ---")
        try:
            start = time.time()
            output = call_model(prompt)
            elapsed = time.time() - start
            print(output)
            print(f"\n[{elapsed:.2f}s, ~{len(prompt.split())} prompt words]")
            results_log[task_key][technique] = {
                "prompt": prompt,
                "output": output,
                "seconds": round(elapsed, 2),
            }
        except Exception as e:
            print(f"ERROR generating: {e}")
            results_log[task_key][technique] = {"error": str(e)}


def main():
    global generator

    parser = argparse.ArgumentParser(description="Compare prompting techniques on a local HF model.")
    parser.add_argument(
        "--task",
        choices=list(TASKS.keys()) + ["all"],
        default="all",
        help="Which task to run (default: all)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model name (default: gpt2). Try gpt2-medium or gpt2-large for a bit more capability.",
    )
    parser.add_argument(
        "--save",
        default="results.json",
        help="Path to save results as JSON (default: results.json)",
    )
    args = parser.parse_args()

    print(f"Loading model '{args.model}' locally (first run downloads it)...")
    generator = pipeline("text-generation", model=args.model)
    print("Model loaded.\n")

    results_log = {}
    task_keys = list(TASKS.keys()) if args.task == "all" else [args.task]

    for task_key in task_keys:
        run_task(task_key, results_log)

    with open(args.save, "w") as f:
        json.dump(results_log, f, indent=2)

    print(f"\n\nAll results saved to {args.save}")
    print("Open STUDY_GUIDE.md alongside this output to interpret what you're seeing.")


if __name__ == "__main__":
    main()