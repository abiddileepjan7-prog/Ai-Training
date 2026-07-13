from transformers import pipeline, set_seed

# The GPT-2 model downloads automatically the first time this script runs.
set_seed(42)
generator = pipeline("text-generation", model="gpt2")

# Task 1: Sentiment analysis
review = "The battery lasts all day, but the camera is poor and the price is too high."

sentiment_prompts = {
    "Zero-shot": f"""Classify this review as Positive, Negative, or Neutral.
Review: "{review}"
Sentiment:""",
    "One-shot": f"""Review: "The headphones sound excellent and the battery lasts all day."
Sentiment: Positive

Review: "{review}"
Sentiment:""",
    "Few-shot": f"""Review: "This product broke after one day."
Sentiment: Negative

Review: "It works as expected, nothing special."
Sentiment: Neutral

Review: "Excellent quality. I love it."
Sentiment: Positive

Review: "{review}"
Sentiment:""",
    "Chain-of-thought": f"""Classify this review as Positive, Negative, or Neutral.
First identify the positive and negative points. Then give the final sentiment.

Review: "{review}"
Reasoning:""",
}

# Task 2: Logical riddle
riddle = "All roses are flowers. Red Rose is a rose. Is Red Rose a flower?"

riddle_prompts = {
    "Zero-shot": f"""Answer yes or no.
Question: {riddle}
Answer:""",
    "One-shot": f"""Question: All cats are animals. Whiskers is a cat. Is Whiskers an animal?
Answer: Yes

Question: {riddle}
Answer:""",
    "Few-shot": f"""Question: All cats are animals. Whiskers is a cat. Is Whiskers an animal?
Answer: Yes

Question: All birds have wings. Kiwi is a bird. Does Kiwi have wings?
Answer: Yes

Question: All cars are vehicles. A bicycle is not a car. Is a bicycle a vehicle?
Answer: No

Question: {riddle}
Answer:""",
    "Chain-of-thought": f"""Answer the question step by step, then give the final answer.
Question: {riddle}
Reasoning:""",
}

for task_name, prompts in [("Sentiment Analysis", sentiment_prompts), ("Logical Riddle", riddle_prompts)]:
    print(f"\n{'=' * 60}\n{task_name}\n{'=' * 60}")

    for technique, prompt in prompts.items():
        result = generator(
            prompt,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=generator.tokenizer.eos_token_id,
        )
        generated_text = result[0]["generated_text"]
        answer = generated_text[len(prompt):].strip()
        print(f"\n--- {technique} ---")
        print(answer)
