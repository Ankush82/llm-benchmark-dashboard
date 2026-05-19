import os
"""
MT-Bench Style Multi-Turn Chat Benchmark — OpenRouter
Tests multi-turn conversation quality across 8 categories.
Uses an LLM judge (via OpenRouter) to score responses 1-10.

Categories (from original MT-Bench paper):
    Writing, Roleplay, Reasoning, Math, Coding,
    Extraction, STEM, Humanities

Each category has:
    - Turn 1: Initial question
    - Turn 2: Follow-up that requires memory of Turn 1

Scoring (LLM-as-judge):
    1-3  : Poor — wrong, incoherent, or ignores the question
    4-6  : Average — partially correct, some issues
    7-8  : Good — correct, clear, mostly complete
    9-10 : Excellent — accurate, insightful, well-structured

Usage:
    python3 mt_bench.py

Requirements:
    pip3 install openai
"""

import re
import time
import json
from datetime import datetime
from openai import OpenAI

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Models being tested
MODELS = {
    "Phi-4":     "microsoft/phi-4",
    "Nemotron":  "nvidia/nemotron-3-super-120b-a12b",
    "Ministral": "mistralai/ministral-14b-2512",
}

# Judge model — scores all responses 1-10
# Using a strong free model as judge
JUDGE_MODEL = "google/gemma-3-27b-it"

# ══════════════════════════════════════════════════════════════════════════════

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ══════════════════════════════════════════════════════════════════════════════
#  MT-BENCH QUESTIONS
#  Each entry has two turns. Turn 2 always references Turn 1.
# ══════════════════════════════════════════════════════════════════════════════

MT_BENCH_QUESTIONS = [

    # ── 1. Writing ─────────────────────────────────────────────────────────────
    {
        "id":       "writing_01",
        "category": "Writing",
        "turn_1":   (
            "Write a short cover letter for a software engineer applying "
            "to a startup. Keep it under 150 words. Make it enthusiastic but professional."
        ),
        "turn_2":   (
            "Now rewrite the same cover letter but for a large enterprise company "
            "like IBM or Accenture. Adjust the tone and emphasis accordingly. "
            "Keep it under 150 words."
        ),
        "judge_criteria_t1": "Quality of writing, tone appropriateness for a startup, enthusiasm, professionalism, staying under 150 words.",
        "judge_criteria_t2": "How well the model adjusted tone for enterprise vs startup. Coherence with Turn 1's content while changing style.",
    },
    {
        "id":       "writing_02",
        "category": "Writing",
        "turn_1":   (
            "Write a one-paragraph product description for a smart water bottle "
            "that tracks hydration and connects to your phone."
        ),
        "turn_2":   (
            "Now rewrite that description as if you were writing for a 10-year-old child. "
            "Make it fun and simple."
        ),
        "judge_criteria_t1": "Clarity, persuasiveness, accuracy of product description, engagement.",
        "judge_criteria_t2": "Age-appropriate language, simplification of technical concepts, fun tone, continuity with the same product.",
    },

    # ── 2. Roleplay ────────────────────────────────────────────────────────────
    {
        "id":       "roleplay_01",
        "category": "Roleplay",
        "turn_1":   (
            "You are a strict but fair job interviewer at a top tech company. "
            "Ask me one challenging behavioural interview question."
        ),
        "turn_2":   (
            "I answered: 'I once missed a major deadline because I underestimated "
            "the complexity of the task. I owned up to it, communicated early, "
            "and we recovered.' "
            "As the interviewer, give me detailed feedback on that answer."
        ),
        "judge_criteria_t1": "Quality of the behavioural question, relevance to tech roles, difficulty level, staying in character.",
        "judge_criteria_t2": "Depth of feedback, staying in interviewer character, constructiveness, referencing the specific answer given.",
    },
    {
        "id":       "roleplay_02",
        "category": "Roleplay",
        "turn_1":   (
            "Act as a doctor explaining to a patient that they have been "
            "diagnosed with Type 2 diabetes. Be empathetic and clear."
        ),
        "turn_2":   (
            "The patient asks: 'Does this mean I can never eat sugar again?' "
            "Respond as the doctor."
        ),
        "judge_criteria_t1": "Empathy, clarity of medical explanation, appropriate bedside manner, accuracy.",
        "judge_criteria_t2": "Medical accuracy, reassurance, practical guidance, maintaining doctor persona, connecting to Turn 1.",
    },

    # ── 3. Reasoning ──────────────────────────────────────────────────────────
    {
        "id":       "reasoning_01",
        "category": "Reasoning",
        "turn_1":   (
            "There are 5 houses in a row, each painted a different colour. "
            "Each house is occupied by a person of a different nationality. "
            "The Norwegian lives in the first house. "
            "The green house is immediately to the left of the white house. "
            "Given only these two clues so far, what can you logically deduce "
            "about the position of the green and white houses?"
        ),
        "turn_2":   (
            "Now add this clue: the Norwegian does not live in the green or white house. "
            "What new deductions can you make? "
            "Show your reasoning step by step."
        ),
        "judge_criteria_t1": "Logical correctness, clarity of deductions, not over-concluding beyond what clues allow.",
        "judge_criteria_t2": "Correctly integrating the new clue, building on Turn 1's deductions, step-by-step logic.",
    },
    {
        "id":       "reasoning_02",
        "category": "Reasoning",
        "turn_1":   (
            "Three colleagues — Alice, Bob, and Carol — each work in a different "
            "department: Engineering, Marketing, and Finance. "
            "Alice does not work in Engineering. "
            "Bob does not work in Marketing. "
            "Who works in which department? Show your reasoning."
        ),
        "turn_2":   (
            "Now suppose Carol gets transferred and a new person, David, joins. "
            "David works in the same department Carol used to work in. "
            "Alice now moves to Engineering. "
            "What department does Bob work in now? Explain your reasoning."
        ),
        "judge_criteria_t1": "Correct logical deduction, clear explanation, proper use of elimination.",
        "judge_criteria_t2": "Correctly updating state from Turn 1, tracking the changes, arriving at correct answer.",
    },

    # ── 4. Math ────────────────────────────────────────────────────────────────
    {
        "id":       "math_01",
        "category": "Math",
        "turn_1":   (
            "Explain the concept of compound interest to someone who has "
            "never studied finance. Use a simple example with actual numbers."
        ),
        "turn_2":   (
            "Using the same example you just gave, calculate how much money "
            "someone would have after 10 years if they invested $5,000 "
            "at 6% annual compound interest. Show your calculation."
        ),
        "judge_criteria_t1": "Clarity of explanation, appropriate example, accuracy of the concept.",
        "judge_criteria_t2": "Mathematical accuracy, using the same example from Turn 1, showing full calculation, correct final answer (~$8,954).",
    },
    {
        "id":       "math_02",
        "category": "Math",
        "turn_1":   (
            "What is the difference between mean, median, and mode? "
            "Give a simple example dataset to illustrate each."
        ),
        "turn_2":   (
            "Using your example dataset from above, if I add the number 100 to the dataset, "
            "how does each of the three measures change? "
            "Which measure is most affected and why?"
        ),
        "judge_criteria_t1": "Correct definitions, clear example dataset, clear illustrations of each measure.",
        "judge_criteria_t2": "Correctly recalculating using the SAME dataset from Turn 1, correct analysis of which measure is most affected (mean).",
    },

    # ── 5. Coding ─────────────────────────────────────────────────────────────
    {
        "id":       "coding_01",
        "category": "Coding",
        "turn_1":   (
            "Write a Python function that takes a list of numbers "
            "and returns the two numbers that sum closest to zero. "
            "Include docstring and example usage."
        ),
        "turn_2":   (
            "Now modify your function to handle edge cases: "
            "empty list, list with one element, and list with all positive numbers. "
            "Show the updated function and explain what you changed."
        ),
        "judge_criteria_t1": "Correctness of the algorithm, code quality, docstring, example usage.",
        "judge_criteria_t2": "Correctly handling all three edge cases mentioned, building on Turn 1's function, clear explanation of changes.",
    },
    {
        "id":       "coding_02",
        "category": "Coding",
        "turn_1":   (
            "Explain what a decorator is in Python. "
            "Write a simple decorator that logs how long a function takes to run."
        ),
        "turn_2":   (
            "Now extend your decorator to also log the function's name "
            "and its arguments each time it's called. "
            "Show updated code and a usage example."
        ),
        "judge_criteria_t1": "Correct explanation, correct decorator implementation, timing accuracy.",
        "judge_criteria_t2": "Correctly extending the SAME decorator from Turn 1, logging name and args correctly, example works.",
    },

    # ── 6. Extraction ──────────────────────────────────────────────────────────
    {
        "id":       "extraction_01",
        "category": "Extraction",
        "turn_1":   (
            "Here is a paragraph:\n\n"
            "'Anthropic, founded in 2021 by former OpenAI researchers including "
            "Dario Amodei and Daniela Amodei, is an AI safety company headquartered "
            "in San Francisco. The company raised $7.3 billion in funding and is "
            "known for its Claude family of AI assistants.'\n\n"
            "Extract all named entities and classify them as: "
            "Person, Organisation, Location, or Number."
        ),
        "turn_2":   (
            "From the entities you extracted, which ones are most likely to change "
            "over time and why? Rank them from most to least likely to change."
        ),
        "judge_criteria_t1": "Completeness and accuracy of extraction, correct classification of all entities.",
        "judge_criteria_t2": "Thoughtful ranking using the SAME entities from Turn 1, good reasoning about what changes over time.",
    },
    {
        "id":       "extraction_02",
        "category": "Extraction",
        "turn_1":   (
            "Read this and answer the question:\n\n"
            "'The meeting on Tuesday had three agenda items: the Q3 budget review, "
            "the new hire onboarding plan, and the product roadmap for 2025. "
            "Sarah presented the budget, which showed a 12% overspend. "
            "The onboarding plan was approved unanimously. "
            "The roadmap discussion was postponed to next week.'\n\n"
            "What were the outcomes of each agenda item?"
        ),
        "turn_2":   (
            "Based on what you extracted, which agenda item requires "
            "the most urgent follow-up action and why? "
            "Suggest one specific next step."
        ),
        "judge_criteria_t1": "Accuracy of extraction, completeness — covers all three items and their outcomes.",
        "judge_criteria_t2": "Reasoning grounded in Turn 1's extracted facts, sensible prioritisation, practical next step.",
    },

    # ── 7. STEM ────────────────────────────────────────────────────────────────
    {
        "id":       "stem_01",
        "category": "STEM",
        "turn_1":   (
            "Explain how CRISPR-Cas9 gene editing works. "
            "Use an analogy to make it accessible to a non-scientist."
        ),
        "turn_2":   (
            "Based on your explanation, what are two potential risks of "
            "using CRISPR on human embryos? "
            "How is the scientific community currently addressing these risks?"
        ),
        "judge_criteria_t1": "Scientific accuracy, quality of the analogy, accessibility of explanation.",
        "judge_criteria_t2": "Scientifically accurate risks, grounded in the Turn 1 explanation, current awareness of the field.",
    },
    {
        "id":       "stem_02",
        "category": "STEM",
        "turn_1":   (
            "What is the difference between supervised, unsupervised, "
            "and reinforcement learning in machine learning? "
            "Give a real-world example of each."
        ),
        "turn_2":   (
            "Of the three types you described, which one would be most "
            "appropriate for training a self-driving car, and why? "
            "What specific challenges would arise?"
        ),
        "judge_criteria_t1": "Correct definitions, distinct real-world examples, clarity.",
        "judge_criteria_t2": "Justified choice referencing Turn 1's definitions, specific and accurate challenges.",
    },

    # ── 8. Humanities ─────────────────────────────────────────────────────────
    {
        "id":       "humanities_01",
        "category": "Humanities",
        "turn_1":   (
            "What were the main causes of World War I? "
            "Summarise in 3-4 key points."
        ),
        "turn_2":   (
            "Of the causes you listed, which one do you think was most significant "
            "and why? Reference at least one historical event or figure to support your argument."
        ),
        "judge_criteria_t1": "Historical accuracy, coverage of main causes (MAIN: Militarism, Alliances, Imperialism, Nationalism), conciseness.",
        "judge_criteria_t2": "Coherent argument referencing Turn 1's causes, historical accuracy, specific evidence cited.",
    },
    {
        "id":       "humanities_02",
        "category": "Humanities",
        "turn_1":   (
            "Explain the philosophical concept of utilitarianism. "
            "Who were its key thinkers and what is the core principle?"
        ),
        "turn_2":   (
            "Using utilitarianism as a framework, how would a strict utilitarian "
            "evaluate the decision to shut down a factory that pollutes a river "
            "but employs 500 people? "
            "What would they weigh up?"
        ),
        "judge_criteria_t1": "Philosophical accuracy, naming Bentham and Mill, correct core principle.",
        "judge_criteria_t2": "Correctly applying the Turn 1 framework to the scenario, balanced analysis of competing utilities.",
    },
]

# ══════════════════════════════════════════════════════════════════════════════
#  LLM JUDGE PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def build_judge_prompt(question, response, criteria, turn_num, conversation_history=None):
    history_text = ""
    if conversation_history:
        history_text = f"\n\nConversation history before this turn:\n{conversation_history}\n"

    return f"""You are an expert judge evaluating the quality of an AI assistant's response.

Score the response on a scale of 1 to 10 using these criteria:
1-3  : Poor — incorrect, incoherent, ignores the question, or completely off-topic
4-5  : Below average — partially addresses the question with notable errors or gaps
6-7  : Good — mostly correct and relevant, minor issues
8-9  : Very good — accurate, clear, well-structured, addresses all aspects
10   : Excellent — outstanding accuracy, insight, and communication

Question (Turn {turn_num}):
{question}
{history_text}
Specific criteria for THIS response:
{criteria}

Response to evaluate:
{response}

Instructions:
- Give a score from 1 to 10
- Give a 2-3 sentence justification
- Output ONLY in this exact format, nothing else:

SCORE: [number]
REASON: [your justification]"""

# ══════════════════════════════════════════════════════════════════════════════
#  API CALLS
# ══════════════════════════════════════════════════════════════════════════════

def call_model(model_id, messages):
    """Call a model with full conversation history."""
    start = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.7,   # slight creativity for chat quality
            max_tokens=800,
        )
        latency = (time.perf_counter() - start) * 1000
        return {
            "success":            True,
            "response":           completion.choices[0].message.content,
            "latency_ms":         round(latency, 1),
            "prompt_tokens":      completion.usage.prompt_tokens,
            "completion_tokens":  completion.usage.completion_tokens,
            "total_tokens":       completion.usage.total_tokens,
            "error":              None,
        }
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return {
            "success": False, "response": "", "latency_ms": round(latency, 1),
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "error": str(e),
        }

def call_judge(question, response, criteria, turn_num, conversation_history=None):
    """Call the judge model to score a response."""
    judge_prompt = build_judge_prompt(
        question, response, criteria, turn_num, conversation_history
    )
    start = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            max_tokens=200,
        )
        raw = completion.choices[0].message.content.strip()

        # Parse score and reason
        score_match  = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', raw)
        reason_match = re.search(r'REASON:\s*(.+)', raw, re.DOTALL)

        score  = float(score_match.group(1))  if score_match  else -1
        reason = reason_match.group(1).strip() if reason_match else raw

        # Clamp to 1-10
        score = max(1.0, min(10.0, score))

        return {
            "success":    True,
            "score":      score,
            "reason":     reason,
            "raw_output": raw,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "error":      None,
        }
    except Exception as e:
        return {
            "success": False, "score": -1, "reason": "",
            "raw_output": "", "latency_ms": 0, "error": str(e),
        }

def call_judge_with_retry(question, response, criteria, turn_num,
                          conversation_history=None, max_retries=2):
    """
    Wraps call_judge with retry logic.
    Retries if score is -1 (parse failure) or suspiciously low (<= 2),
    which usually means the judge itself got confused rather than
    the model giving a genuinely bad response.
    """
    for attempt in range(1, max_retries + 1):
        result = call_judge(question, response, criteria, turn_num, conversation_history)
        if result["success"] and result["score"] > 2:
            return result
        if attempt < max_retries:
            print(f" [judge retry {attempt}]", end=" ", flush=True)
            time.sleep(2.0)
    # Return last result even if still suspicious
    return result

# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark():
    all_results = {}

    for model_label, model_id in MODELS.items():
        print(f"\n{'═'*68}")
        print(f"  Model: {model_label}  ({model_id})")
        print(f"{'═'*68}")

        model_results = []

        for q in MT_BENCH_QUESTIONS:
            print(f"\n  [{q['id']}] {q['category']}")

            # ── Turn 1 ──────────────────────────────────────────────────────
            print(f"    Turn 1 ...", end=" ", flush=True)

            messages_t1 = [{"role": "user", "content": q["turn_1"]}]
            result_t1   = call_model(model_id, messages_t1)

            if result_t1["success"]:
                # Judge Turn 1 — with retry if score looks suspicious
                time.sleep(0.5)
                judge_t1 = call_judge_with_retry(
                    question   = q["turn_1"],
                    response   = result_t1["response"],
                    criteria   = q["judge_criteria_t1"],
                    turn_num   = 1,
                )
                score_t1 = judge_t1["score"]
                print(f"Score: {score_t1}/10  ({result_t1['latency_ms']}ms, {result_t1['total_tokens']} tok)")
            else:
                judge_t1 = {"success": False, "score": -1, "reason": "", "error": result_t1["error"]}
                score_t1 = -1
                print(f"ERROR: {result_t1['error']}")

            time.sleep(1.0)

            # ── Turn 2 ──────────────────────────────────────────────────────
            print(f"    Turn 2 ...", end=" ", flush=True)

            # Safe fallback if Turn 1 response was None
            t1_response_text = result_t1["response"] or ""

            # Send full conversation history so model remembers Turn 1
            messages_t2 = [
                {"role": "user",      "content": q["turn_1"]},
                {"role": "assistant", "content": t1_response_text},
                {"role": "user",      "content": q["turn_2"]},
            ]
            result_t2 = call_model(model_id, messages_t2)

            if result_t2["success"]:
                # Provide Turn 1 context to judge — safely sliced
                conv_history = (
                    f"Turn 1 Question: {q['turn_1']}\n"
                    f"Turn 1 Response: {t1_response_text[:400]}..."
                )
                time.sleep(0.5)
                judge_t2 = call_judge_with_retry(
                    question             = q["turn_2"],
                    response             = result_t2["response"],
                    criteria             = q["judge_criteria_t2"],
                    turn_num             = 2,
                    conversation_history = conv_history,
                )
                score_t2 = judge_t2["score"]
                print(f"Score: {score_t2}/10  ({result_t2['latency_ms']}ms, {result_t2['total_tokens']} tok)")
            else:
                judge_t2 = {"success": False, "score": -1, "reason": "", "error": result_t2["error"]}
                score_t2 = -1
                print(f"ERROR: {result_t2['error']}")

            avg_score = round((score_t1 + score_t2) / 2, 1) if score_t1 > 0 and score_t2 > 0 else -1

            model_results.append({
                "question_id":       q["id"],
                "category":          q["category"],
                "turn_1_question":   q["turn_1"],
                "turn_2_question":   q["turn_2"],
                "turn_1_response":   t1_response_text,
                "turn_2_response":   result_t2["response"] if result_t2["success"] else "",
                "turn_1_score":      score_t1,
                "turn_2_score":      score_t2,
                "avg_score":         avg_score,
                "turn_1_reason":     judge_t1.get("reason", ""),
                "turn_2_reason":     judge_t2.get("reason", ""),
                "turn_1_latency_ms": result_t1["latency_ms"],
                "turn_2_latency_ms": result_t2["latency_ms"] if result_t2["success"] else 0,
                "turn_1_tokens":     result_t1["total_tokens"],
                "turn_2_tokens":     result_t2["total_tokens"] if result_t2["success"] else 0,
            })

            time.sleep(1.5)

        all_results[model_label] = {
            "model_id": model_id,
            "results":  model_results,
        }

    return all_results

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary(all_results):
    summary = {}
    for model_label, data in all_results.items():
        results = data["results"]
        valid   = [r for r in results if r["avg_score"] > 0]

        # Overall
        avg_score   = sum(r["avg_score"]         for r in valid) / len(valid) if valid else 0
        avg_t1      = sum(r["turn_1_score"]      for r in valid) / len(valid) if valid else 0
        avg_t2      = sum(r["turn_2_score"]       for r in valid) / len(valid) if valid else 0
        avg_latency = sum(r["turn_1_latency_ms"] + r["turn_2_latency_ms"] for r in results) / len(results) if results else 0
        avg_tokens  = sum(r["turn_1_tokens"]     + r["turn_2_tokens"]     for r in results) / len(results) if results else 0

        # Per category
        by_category = {}
        for r in valid:
            cat = r["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r["avg_score"])

        cat_avgs = {cat: round(sum(scores)/len(scores), 1) for cat, scores in by_category.items()}

        # Turn 1 vs Turn 2 drop-off
        t2_drop = round(avg_t1 - avg_t2, 2)

        summary[model_label] = {
            "avg_score":      round(avg_score, 2),
            "avg_turn_1":     round(avg_t1, 2),
            "avg_turn_2":     round(avg_t2, 2),
            "turn2_dropoff":  t2_drop,
            "avg_latency_ms": round(avg_latency, 1),
            "avg_tokens":     round(avg_tokens, 1),
            "by_category":    cat_avgs,
        }
    return summary

# ══════════════════════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_report(all_results, summary):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"mt_bench_{timestamp}.txt"
    D = "=" * 72
    T = "-" * 72

    categories = ["Writing", "Roleplay", "Reasoning", "Math",
                  "Coding", "Extraction", "STEM", "Humanities"]

    with open(filename, "w", encoding="utf-8") as f:

        # Header
        f.write(D + "\n")
        f.write("  MT-BENCH STYLE MULTI-TURN CHAT BENCHMARK\n")
        f.write(D + "\n")
        f.write(f"  Date         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Models tested: {', '.join(MODELS.keys())}\n")
        f.write(f"  Judge model  : {JUDGE_MODEL}\n")
        f.write(f"  Questions    : {len(MT_BENCH_QUESTIONS)} (2 turns each = {len(MT_BENCH_QUESTIONS)*2} total responses)\n")
        f.write(f"  Scoring      : 1-10 per turn, averaged\n")
        f.write(D + "\n\n")

        # Summary table
        f.write("OVERALL SCORES\n")
        f.write(T + "\n")
        f.write(f"  {'Model':<18} {'Overall':>9} {'Turn 1':>8} {'Turn 2':>8} {'T2 Drop':>9} {'Avg Tokens':>12}\n")
        f.write(T + "\n")
        for label, s in summary.items():
            drop_str = f"-{s['turn2_dropoff']}" if s["turn2_dropoff"] > 0 else f"+{abs(s['turn2_dropoff'])}"
            f.write(
                f"  {label:<18}"
                f"  {s['avg_score']:>7.2f}/10"
                f"  {s['avg_turn_1']:>6.2f}/10"
                f"  {s['avg_turn_2']:>6.2f}/10"
                f"  {drop_str:>8}"
                f"  {s['avg_tokens']:>10.0f}\n"
            )
        f.write(T + "\n\n")

        # Best model per category
        f.write("SCORES BY CATEGORY\n")
        f.write(T + "\n")
        header = f"  {'Category':<16}"
        for label in MODELS:
            header += f"  {label:>12}"
        header += f"  {'Best':>10}"
        f.write(header + "\n")
        f.write(T + "\n")

        for cat in categories:
            row    = f"  {cat:<16}"
            scores = {}
            for label, s in summary.items():
                score = s["by_category"].get(cat, 0)
                scores[label] = score
                row += f"  {score:>11.1f}"
            best = max(scores, key=scores.get)
            row += f"  {best:>10}"
            f.write(row + "\n")
        f.write(T + "\n\n")

        # Key insight: Turn 2 drop-off
        f.write("MULTI-TURN MEMORY ANALYSIS\n")
        f.write(T + "\n")
        f.write("  (Turn 2 Drop-off = how much score fell from Turn 1 to Turn 2)\n")
        f.write("  A smaller drop means the model better retains context across turns.\n\n")
        for label, s in summary.items():
            drop = s["turn2_dropoff"]
            assessment = (
                "Excellent context retention" if drop <= 0.3 else
                "Good context retention"      if drop <= 0.7 else
                "Moderate drop-off"           if drop <= 1.5 else
                "Significant context loss"
            )
            f.write(f"  {label:<18}: {drop:+.2f} points  → {assessment}\n")
        f.write(T + "\n\n")

        # Per-model detailed breakdown
        for model_label, data in all_results.items():
            f.write(D + "\n")
            f.write(f"  MODEL: {model_label}   ({MODELS[model_label]})\n")
            s = summary[model_label]
            f.write(f"  Overall score    : {s['avg_score']}/10\n")
            f.write(f"  Turn 1 avg       : {s['avg_turn_1']}/10\n")
            f.write(f"  Turn 2 avg       : {s['avg_turn_2']}/10\n")
            f.write(f"  Turn 2 drop-off  : {s['turn2_dropoff']:+.2f}\n")
            f.write(f"  Avg tokens/q     : {s['avg_tokens']:.0f}\n")
            f.write(f"  Avg latency/q    : {s['avg_latency_ms']:.0f}ms\n\n")

            f.write("  Category scores:\n")
            for cat, score in s["by_category"].items():
                bar = "█" * int(score) + "░" * (10 - int(score))
                f.write(f"    {cat:<14} [{bar}] {score:.1f}/10\n")
            f.write("\n")

            f.write(T + "\n")
            f.write("  Question-by-question results:\n")
            f.write(T + "\n\n")

            for r in data["results"]:
                f.write(f"  [{r['question_id']}] {r['category']}\n")
                f.write(f"  Turn 1 Score: {r['turn_1_score']}/10\n")
                f.write(f"  Turn 1 Reason: {r['turn_1_reason']}\n\n")
                f.write(f"  Turn 1 Question:\n    {r['turn_1_question']}\n\n")
                f.write(f"  Turn 1 Response:\n")
                for line in r["turn_1_response"].split("\n"):
                    f.write(f"    {line}\n")
                f.write(f"\n  Turn 2 Score: {r['turn_2_score']}/10\n")
                f.write(f"  Turn 2 Reason: {r['turn_2_reason']}\n\n")
                f.write(f"  Turn 2 Question:\n    {r['turn_2_question']}\n\n")
                f.write(f"  Turn 2 Response:\n")
                for line in r["turn_2_response"].split("\n"):
                    f.write(f"    {line}\n")
                f.write(f"\n  Avg Score: {r['avg_score']}/10\n")
                f.write(f"  Tokens: T1={r['turn_1_tokens']} T2={r['turn_2_tokens']}\n")
                f.write(f"  Latency: T1={r['turn_1_latency_ms']}ms T2={r['turn_2_latency_ms']}ms\n")
                f.write("\n" + T + "\n\n")

        f.write(D + "\n  END OF REPORT\n" + D + "\n")

    return filename

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    total_api_calls = (len(MT_BENCH_QUESTIONS) * 2 * len(MODELS)) + (len(MT_BENCH_QUESTIONS) * 2)
    print("=" * 68)
    print("  MT-Bench Style Chat Benchmark — OpenRouter")
    print("=" * 68)
    print(f"  Models tested  : {', '.join(MODELS.keys())}")
    print(f"  Judge model    : {JUDGE_MODEL}")
    print(f"  Questions      : {len(MT_BENCH_QUESTIONS)} × 2 turns")
    print(f"  Total API calls: ~{total_api_calls}")
    print(f"  Categories     : Writing, Roleplay, Reasoning, Math,")
    print(f"                   Coding, Extraction, STEM, Humanities")
    print("=" * 68)

    all_results = run_benchmark()
    summary     = compute_summary(all_results)
    report_file = write_report(all_results, summary)

    # Print final scores to console
    print(f"\n{'='*68}")
    print("  FINAL SCORES")
    print(f"{'='*68}")
    for label, s in summary.items():
        print(f"  {label:<18}: {s['avg_score']}/10  (T1={s['avg_turn_1']}  T2={s['avg_turn_2']}  drop={s['turn2_dropoff']:+.2f})")
    print(f"\n  Report saved → {report_file}")
    print(f"{'='*68}")