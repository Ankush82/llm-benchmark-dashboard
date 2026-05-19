import os
"""
Reasoning Recovery Benchmark v2 — Harder Traps
Novel problems engineered to trip up LLMs.

What makes these harder than v1:
  - Multi-layered traps (multiple wrong paths)
  - Novel phrasings not seen in training data
  - Problems requiring careful re-reading
  - Contradictions hidden in plain sight
  - False intermediate results that look correct
  - Unusual number combinations that defeat pattern matching

Usage:
    python3 reasoning_recovery_v2.py

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

MODELS = {
    "Phi-4":     "microsoft/phi-4",
    "Nemotron":  "nvidia/nemotron-3-super-120b-a12b",
    "Ministral": "mistralai/ministral-14b-2512",
}

# ══════════════════════════════════════════════════════════════════════════════

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

CORRECTION_SIGNALS = [
    "wait", "actually", "let me reconsider", "i made an error",
    "i made a mistake", "that's wrong", "that is wrong",
    "let me recalculate", "let me redo", "i was wrong",
    "correction:", "on second thought", "i need to reconsider",
    "let me re-examine", "i realize", "i see my mistake",
    "my mistake", "i got that wrong", "re-reading the problem",
    "let me re-read", "hold on", "oops", "not quite",
    "i need to re-read", "i misread", "i overlooked",
    "careful", "let me check", "double check", "rechecking",
]

# ══════════════════════════════════════════════════════════════════════════════
#  TRAP PROBLEMS v2 — Harder, novel, multi-layered
# ══════════════════════════════════════════════════════════════════════════════

TRAP_PROBLEMS = [

    # ── CATEGORY 1: Compounding rate traps ────────────────────────────────────
    # Models conflate simple and compound rates
    {
        "id": "v2_01",
        "category": "Compounding Rate Trap",
        "question": (
            "A bacteria colony doubles every 3 hours. "
            "At 12:00 noon, the colony fills exactly half the petri dish. "
            "At what time will the dish be completely full? "
            "Show your reasoning."
        ),
        "trap": "Models calculate forward from noon instead of realising: if it doubles every 3 hours and is half full at noon, it will be full in exactly ONE more doubling period — 3pm.",
        "correct": "3:00 pm",
        "explanation": "Half full at noon + one doubling = full at 3pm. Trap: models try to calculate backwards or count multiple doublings forward.",
        "difficulty": "Medium",
    },
    {
        "id": "v2_02",
        "category": "Compounding Rate Trap",
        "question": (
            "A car depreciates by 20% each year. "
            "After how many years will the car be worth less than 40% of its original value? "
            "Show your calculations year by year."
        ),
        "trap": "Models apply 20% per year linearly (thinking 2 years = 40% loss) rather than compounding. After year 1: 80%, year 2: 64%, year 3: 51.2%, year 4: 40.96% — so the answer is 5 years not 3.",
        "correct": "5",
        "explanation": "Year 1: 80%, Y2: 64%, Y3: 51.2%, Y4: 40.96%, Y5: 32.77%. Falls below 40% in year 5. Trap: linear thinking gives wrong answer of 3.",
        "difficulty": "Hard",
    },

    # ── CATEGORY 2: Embedded contradiction traps ───────────────────────────────
    # The problem contains a subtle internal contradiction or impossible condition
    {
        "id": "v2_03",
        "category": "Embedded Contradiction Trap",
        "question": (
            "A ladder is leaning against a wall. "
            "The base of the ladder is 6 feet from the wall. "
            "The top of the ladder reaches 8 feet up the wall. "
            "The ladder is 11 feet long. "
            "What angle does the ladder make with the ground? "
            "Show your working."
        ),
        "trap": "Models calculate the angle using 6 and 8 without checking: a 6-8 ladder should be 10 feet (Pythagoras), not 11. The problem has an internal inconsistency. A correct model should flag this.",
        "correct": "impossible",
        "explanation": "6² + 8² = 36 + 64 = 100 = 10². A ladder with base 6, height 8 must be 10ft, not 11ft. The problem is geometrically impossible. Models should catch this.",
        "difficulty": "Hard",
    },
    {
        "id": "v2_04",
        "category": "Embedded Contradiction Trap",
        "question": (
            "In a survey of 100 people: "
            "60 people like coffee. "
            "50 people like tea. "
            "30 people like neither coffee nor tea. "
            "How many people like both coffee and tea? "
            "Show your working."
        ),
        "trap": "Models jump to inclusion-exclusion: Both = 60+50-70 = 40. But wait — 30 like neither, so 70 like at least one. Both = 60+50-70 = 40. This is actually correct but models often get confused with the 'neither' framing and use 100 instead of 70.",
        "correct": "40",
        "explanation": "People who like at least one = 100-30 = 70. By inclusion-exclusion: 60+50-both=70, so both=40. Trap: using 100 instead of 70 gives wrong answer of 10.",
        "difficulty": "Medium",
    },

    # ── CATEGORY 3: Rate and work traps ───────────────────────────────────────
    # Novel combined rate problems with hidden wrinkles
    {
        "id": "v2_05",
        "category": "Rate and Work Trap",
        "question": (
            "Pipe A fills a tank in 4 hours. "
            "Pipe B fills the same tank in 6 hours. "
            "Pipe C drains the tank in 3 hours. "
            "If all three pipes are open simultaneously, "
            "will the tank ever be full? "
            "If yes, how long does it take? "
            "If no, explain why."
        ),
        "trap": "Models often just add rates: 1/4 + 1/6 - 1/3 = 3/12 + 2/12 - 4/12 = 1/12. Negative combined rate means the tank drains. Models often get the arithmetic wrong or give a positive time.",
        "correct": "no",
        "explanation": "Net rate = 1/4 + 1/6 - 1/3 = 3/12 + 2/12 - 4/12 = 1/12. Wait — 1/12 is positive! Tank fills in 12 hours. Trap: many models miscalculate as negative. Correct answer is YES, 12 hours.",
        "correct": "12",
        "difficulty": "Hard",
    },
    {
        "id": "v2_06",
        "category": "Rate and Work Trap",
        "question": (
            "Alice can paint a house in 5 days. "
            "Bob can paint the same house in 7 days. "
            "They start painting together, but after 2 days "
            "Alice leaves and Bob finishes alone. "
            "How many total days does it take to paint the house? "
            "Round to 1 decimal place."
        ),
        "trap": "Models forget that the 'total days' includes the 2 days Alice worked. They often give just Bob's remaining time.",
        "correct": "6.4",
        "explanation": "In 2 days together: 2*(1/5+1/7) = 2*12/35 = 24/35 done. Remaining: 11/35. Bob alone: (11/35)/(1/7) = 11*7/35 = 77/35 = 2.2 days. Wait — let me recalculate. 11/35 * 7 = 77/35 = 2.2 days. Total = 2 + 2.2 = 4.2 days.",
        "correct": "4.2",
        "difficulty": "Hard",
    },

    # ── CATEGORY 4: Sequence and pattern traps ─────────────────────────────────
    # Models pattern-match to wrong sequences
    {
        "id": "v2_07",
        "category": "Sequence and Pattern Trap",
        "question": (
            "What is the next number in this sequence: "
            "1, 2, 3, 5, 8, 13, 21, 34, 56 ... "
            "Show your reasoning."
        ),
        "trap": "This looks like Fibonacci (where each = sum of previous two) but 56 is wrong — it should be 55 (21+34=55 not 56). Models trained on Fibonacci will say 89 or 90 without catching the error in the sequence.",
        "correct": "there is an error in the sequence",
        "explanation": "21+34=55, not 56. The sequence has an error at position 9. A correct model should flag this. If forced to continue the broken pattern: 34+56=90.",
        "difficulty": "Very Hard",
    },
    {
        "id": "v2_08",
        "category": "Sequence and Pattern Trap",
        "question": (
            "A snail climbs 3 feet up a 30-foot well during the day "
            "and slides back 2 feet at night. "
            "On which day does the snail first reach the top? "
            "Show your day-by-day reasoning."
        ),
        "trap": "Models often say 30 days (30-2=28 net... wait no). The snail climbs 1 foot net per day, but on the final day it reaches 30 before sliding. Net per day = 1ft. After 27 days it is at 27ft. Day 28: climbs to 30ft and escapes — does not slide back.",
        "correct": "28",
        "explanation": "Net 1ft/day. After 27 days = 27ft. Day 28: climbs 3ft to 30ft, reaches top, done. Trap: models say 30 days or 27 days by forgetting the snail doesn't slide on the day it escapes.",
        "difficulty": "Medium",
    },

    # ── CATEGORY 5: Percentage and proportion traps ────────────────────────────
    # Novel percentage problems with counter-intuitive answers
    {
        "id": "v2_09",
        "category": "Percentage and Proportion Trap",
        "question": (
            "A shop increases its price by 25% on Monday. "
            "On Tuesday it decreases the new price by 25%. "
            "On Wednesday it increases the Tuesday price by 25% again. "
            "On Thursday it decreases the Wednesday price by 25%. "
            "If the original price was $100, what is the final price on Thursday? "
            "Show all steps."
        ),
        "trap": "Models think +25% -25% cancels out. It doesn't. Each cycle: 100 * 1.25 * 0.75 = 93.75. Two cycles: 100 * (1.25*0.75)² = 100 * 0.9375² = 87.89.",
        "correct": "87.89",
        "explanation": "Mon: 125, Tue: 93.75, Wed: 117.19, Thu: 87.89. Trap: percentage increases and decreases are NOT symmetric.",
        "difficulty": "Hard",
    },
    {
        "id": "v2_10",
        "category": "Percentage and Proportion Trap",
        "question": (
            "In a room, 70% of people are wearing hats. "
            "Of those wearing hats, 40% are also wearing glasses. "
            "Of those NOT wearing hats, 80% are wearing glasses. "
            "What percentage of the total people in the room are wearing glasses? "
            "Show your working."
        ),
        "trap": "Models average the percentages: (40+80)/2 = 60%. Wrong. Need to weight by proportion: 0.7*0.4 + 0.3*0.8 = 0.28 + 0.24 = 0.52 = 52%.",
        "correct": "52",
        "explanation": "Weighted average: (0.7 × 40%) + (0.3 × 80%) = 28% + 24% = 52%. Trap: simple average of 40% and 80% gives wrong 60%.",
        "difficulty": "Hard",
    },

    # ── CATEGORY 6: Logic and language traps ──────────────────────────────────
    # Careful reading required — the language is the trap
    {
        "id": "v2_11",
        "category": "Language and Logic Trap",
        "question": (
            "A box contains red and blue marbles. "
            "The ratio of red to blue is 3:2. "
            "If 10 red marbles are removed, "
            "the ratio becomes 1:2. "
            "How many blue marbles are in the box? "
            "Show your working."
        ),
        "trap": "Models set up wrong equation. Let red=3x, blue=2x. After removal: (3x-10)/(2x) = 1/2. Cross multiply: 6x-20=2x. 4x=20. x=5. Red=15, Blue=10. But check: (15-10)/10 = 5/10 = 1/2 ✓",
        "correct": "10",
        "explanation": "3x-10 / 2x = 1/2 → 6x-20=2x → x=5. Blue=2x=10. Trap: models often set ratio incorrectly or make algebra errors.",
        "difficulty": "Medium",
    },
    {
        "id": "v2_12",
        "category": "Language and Logic Trap",
        "question": (
            "Every student in a class passed at least one of two exams: Maths or English. "
            "13 students passed Maths. "
            "11 students passed English. "
            "4 students passed both. "
            "How many students are in the class? "
            "Now — if one more student joins who failed BOTH exams, "
            "what fraction of the class passed at least one exam? "
            "Show all working."
        ),
        "trap": "Two-part problem. Part 1: 13+11-4=20 students. Part 2: models forget to add the new student to total — they keep 20 as denominator. Correct: 20/(20+1) = 20/21.",
        "correct": "20/21",
        "explanation": "Part 1: 20 students by inclusion-exclusion. Part 2: new total=21, still 20 passed at least one exam. Fraction = 20/21. Trap: using 20 as denominator gives 1 (100%) which is wrong.",
        "difficulty": "Hard",
    },

    # ── CATEGORY 7: Spatial and physical reasoning traps ──────────────────────
    {
        "id": "v2_13",
        "category": "Spatial Reasoning Trap",
        "question": (
            "A cube has a surface area of 150 square centimetres. "
            "What is the length of its diagonal from one corner "
            "to the opposite corner? "
            "Round to 2 decimal places. "
            "Show your working."
        ),
        "trap": "Models often calculate edge diagonal (√2 × side) instead of space diagonal (√3 × side). Surface area = 6s² = 150, so s²=25, s=5. Space diagonal = 5√3 = 8.66.",
        "correct": "8.66",
        "explanation": "s=5. Space diagonal = s√3 = 5×1.732 = 8.66cm. Trap: edge diagonal = 5√2 = 7.07 is a common wrong answer.",
        "difficulty": "Medium",
    },
    {
        "id": "v2_14",
        "category": "Spatial Reasoning Trap",
        "question": (
            "You fold a piece of paper in half 7 times. "
            "The original paper is 0.1mm thick. "
            "How thick is the folded paper in centimetres? "
            "Show your calculations."
        ),
        "trap": "Models calculate 0.1 × 7 = 0.7mm (linear) instead of 0.1 × 2^7 = 12.8mm = 1.28cm (exponential).",
        "correct": "1.28",
        "explanation": "Each fold doubles thickness. 0.1mm × 2^7 = 0.1 × 128 = 12.8mm = 1.28cm. Trap: linear thinking gives 0.07cm.",
        "difficulty": "Medium",
    },

    # ── CATEGORY 8: Trick questions that are simpler than they look ────────────
    {
        "id": "v2_15",
        "category": "Overcomplexity Trap",
        "question": (
            "A train leaves Station A at 9:00am travelling at 80 km/h toward Station B. "
            "Another train leaves Station B at 9:00am travelling at 120 km/h toward Station A. "
            "The distance between Station A and B is 500 km. "
            "A fly starts at Station A at 9:00am and flies back and forth between the two trains "
            "at 200 km/h until the trains collide. "
            "How far does the fly travel in total? "
            "Show your reasoning."
        ),
        "trap": "Models attempt to calculate each leg of the fly's journey (infinite series). The trick is: trains close at 200km/h combined speed, meet in 500/200 = 2.5 hours. Fly travels 200 × 2.5 = 500km.",
        "correct": "500",
        "explanation": "Trains approach at 80+120=200km/h. Time to meet = 500/200 = 2.5 hours. Fly speed × time = 200 × 2.5 = 500km. Trap: attempting the infinite series instead of the elegant shortcut.",
        "difficulty": "Hard",
    },
]

# ── API Call ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise and careful problem solver. "
    "Think through problems step by step showing all working. "
    "Check your assumptions and re-read the problem carefully before answering. "
    "If you notice an inconsistency or error in your reasoning at any point, "
    "say so explicitly and correct yourself."
)

def call_model(model_id, question):
    start = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            temperature=0,
            max_tokens=1500,   # more tokens — harder problems need more space
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

# ── Recovery Analysis ──────────────────────────────────────────────────────────

def find_correction_signals(response_text):
    found = []
    text_lower = response_text.lower()
    for signal in CORRECTION_SIGNALS:
        pos = 0
        while True:
            idx = text_lower.find(signal, pos)
            if idx == -1:
                break
            word_num = len(response_text[:idx].split())
            found.append({"signal": signal, "char_pos": idx, "word_num": word_num})
            pos = idx + 1
    found.sort(key=lambda x: x["char_pos"])
    return found

def check_correct_answer(response_text, correct_answer):
    text_lower   = response_text.lower()
    correct_lower = correct_answer.lower().strip()

    # String answers (impossible, no, yes, etc.)
    if not correct_lower.replace(".", "").replace("-", "").replace("/", "").isdigit():
        return correct_lower in text_lower

    # Fraction answers like "20/21"
    if "/" in correct_lower:
        return correct_lower in text_lower

    try:
        correct_float = float(correct_lower)
        patterns = [
            correct_lower,
            f"{correct_float:.2f}",
            f"{correct_float:.1f}",
            f"{int(correct_float)}" if correct_float == int(correct_float) else None,
        ]
        return any(p and p in text_lower for p in patterns)
    except ValueError:
        return correct_lower in text_lower

def classify_recovery(is_correct, signals_found):
    has_signal = len(signals_found) > 0
    if is_correct and not has_signal:
        return "correct_no_trap"
    elif is_correct and has_signal:
        return "correct_recovered"
    elif not is_correct and has_signal:
        return "false_recovery"
    else:
        return "wrong_no_signal"

def analyse_response(response_text, correct_answer):
    signals       = find_correction_signals(response_text)
    is_correct    = check_correct_answer(response_text, correct_answer)
    recovery_type = classify_recovery(is_correct, signals)

    first_signal_pos   = signals[0]["char_pos"] if signals else None
    total_words        = len(response_text.split())
    recovery_pos_pct   = (
        round(signals[0]["word_num"] / total_words * 100, 1)
        if signals and total_words else None
    )
    tokens_before_corr = round(first_signal_pos / 4) if first_signal_pos else 0

    return {
        "is_correct":               is_correct,
        "recovery_type":            recovery_type,
        "signals_found":            signals,
        "num_signals":              len(signals),
        "first_signal":             signals[0]["signal"] if signals else None,
        "tokens_before_correction": tokens_before_corr,
        "recovery_position_pct":    recovery_pos_pct,
        "total_words":              total_words,
    }

# ── Runner ─────────────────────────────────────────────────────────────────────

def run_benchmark():
    all_results = {}

    for model_label, model_id in MODELS.items():
        print(f"\n{'═'*68}")
        print(f"  Model: {model_label}  ({model_id})")
        print(f"{'═'*68}")

        model_results = []

        for prob in TRAP_PROBLEMS:
            diff = prob.get("difficulty", "")
            print(f"\n  [{prob['id']}] {prob['category']}  [{diff}]")
            print(f"  Q: {prob['question'][:90]}...")

            result   = call_model(model_id, prob["question"])
            analysis = analyse_response(result["response"], prob["correct"])

            record = {
                "problem_id":               prob["id"],
                "category":                 prob["category"],
                "difficulty":               prob.get("difficulty", ""),
                "question":                 prob["question"],
                "correct_answer":           prob["correct"],
                "trap_description":         prob["trap"],
                "model_response":           result["response"],
                "latency_ms":               result["latency_ms"],
                "prompt_tokens":            result["prompt_tokens"],
                "completion_tokens":        result["completion_tokens"],
                "total_tokens":             result["total_tokens"],
                **analysis,
            }
            model_results.append(record)

            status   = "✓" if analysis["is_correct"] else "✗"
            recovery = analysis["recovery_type"].replace("_", " ").upper()
            signals  = [s["signal"] for s in analysis["signals_found"]]

            print(f"  Answer correct  : {status}  |  Recovery: {recovery}")
            if signals:
                print(f"  Signal words    : {', '.join(signals[:3])}")
                print(f"  Caught at       : ~word {analysis['signals_found'][0]['word_num']} of {analysis['total_words']} ({analysis['recovery_position_pct']}%)")
            print(f"  Latency: {result['latency_ms']}ms  |  Tokens: {result['total_tokens']}")

            time.sleep(1.5)

        all_results[model_label] = {"model_id": model_id, "results": model_results}

    return all_results

# ── Summary ────────────────────────────────────────────────────────────────────

def compute_summary(all_results):
    summary = {}
    for model_label, data in all_results.items():
        results = data["results"]
        n = len(results)

        correct         = sum(1 for r in results if r["is_correct"])
        recovered       = sum(1 for r in results if r["recovery_type"] == "correct_recovered")
        false_recovery  = sum(1 for r in results if r["recovery_type"] == "false_recovery")
        wrong_no_signal = sum(1 for r in results if r["recovery_type"] == "wrong_no_signal")
        correct_clean   = sum(1 for r in results if r["recovery_type"] == "correct_no_trap")

        avg_latency = sum(r["latency_ms"]   for r in results) / n
        avg_tokens  = sum(r["total_tokens"] for r in results) / n

        recoveries      = [r for r in results if r["recovery_position_pct"] is not None]
        avg_rec_pos     = (
            sum(r["recovery_position_pct"] for r in recoveries) / len(recoveries)
            if recoveries else None
        )

        # By difficulty
        by_difficulty = {}
        for r in results:
            d = r.get("difficulty", "Unknown")
            if d not in by_difficulty:
                by_difficulty[d] = {"correct": 0, "total": 0}
            by_difficulty[d]["total"] += 1
            if r["is_correct"]:
                by_difficulty[d]["correct"] += 1

        # By category
        by_category = {}
        for r in results:
            cat = r["category"]
            if cat not in by_category:
                by_category[cat] = {"correct": 0, "total": 0}
            by_category[cat]["total"] += 1
            if r["is_correct"]:
                by_category[cat]["correct"] += 1

        summary[model_label] = {
            "total":            n,
            "correct":          correct,
            "accuracy_pct":     round(correct / n * 100, 1),
            "correct_clean":    correct_clean,
            "recovered":        recovered,
            "false_recovery":   false_recovery,
            "wrong_no_signal":  wrong_no_signal,
            "avg_latency_ms":   round(avg_latency, 1),
            "avg_tokens":       round(avg_tokens, 1),
            "avg_rec_pos_pct":  round(avg_rec_pos, 1) if avg_rec_pos else "N/A",
            "by_difficulty":    by_difficulty,
            "by_category":      by_category,
        }
    return summary

# ── Report ─────────────────────────────────────────────────────────────────────

def write_report(all_results, summary):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"reasoning_recovery_v2_{timestamp}.txt"
    D = "=" * 72
    T = "-" * 72

    with open(filename, "w", encoding="utf-8") as f:

        f.write(D + "\n")
        f.write("  REASONING RECOVERY BENCHMARK v2 — HARDER TRAPS\n")
        f.write(D + "\n")
        f.write(f"  Date        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Models      : {', '.join(MODELS.keys())}\n")
        f.write(f"  Problems    : {len(TRAP_PROBLEMS)}\n")
        f.write(f"  Difficulty  : Medium / Hard / Very Hard\n")
        f.write(D + "\n\n")

        # Summary comparison
        f.write("SUMMARY COMPARISON\n")
        f.write(T + "\n")
        f.write(f"  {'Model':<18} {'Accuracy':>10} {'Clean':>7} {'Recovered':>11} {'False':>7} {'Missed':>8} {'Avg Tok':>9}\n")
        f.write(T + "\n")
        for label, s in summary.items():
            f.write(
                f"  {label:<18} {s['accuracy_pct']:>9.1f}%"
                f" {s['correct_clean']:>7}"
                f" {s['recovered']:>11}"
                f" {s['false_recovery']:>7}"
                f" {s['wrong_no_signal']:>8}"
                f" {s['avg_tokens']:>9.1f}\n"
            )
        f.write(T + "\n\n")

        # Per-model detail
        for label, s in summary.items():
            f.write(D + "\n")
            f.write(f"  MODEL: {label}   ({MODELS[label]})\n")
            f.write(T + "\n")
            f.write(f"  Overall accuracy         : {s['correct']}/{s['total']} ({s['accuracy_pct']}%)\n")
            f.write(f"  Solved cleanly           : {s['correct_clean']}\n")
            f.write(f"  Self-corrected (success) : {s['recovered']}\n")
            f.write(f"  False recovery           : {s['false_recovery']}\n")
            f.write(f"  Wrong, never noticed     : {s['wrong_no_signal']}\n")
            f.write(f"  Avg recovery position    : {s['avg_rec_pos_pct']}% through response\n")
            f.write(f"  Avg latency              : {s['avg_latency_ms']} ms\n")
            f.write(f"  Avg tokens               : {s['avg_tokens']}\n\n")

            f.write("  By Difficulty:\n")
            for diff in ["Medium", "Hard", "Very Hard"]:
                stats = s["by_difficulty"].get(diff, {"correct": 0, "total": 0})
                if stats["total"] > 0:
                    acc = round(stats["correct"] / stats["total"] * 100, 1)
                    f.write(f"    {diff:<12} {stats['correct']}/{stats['total']} ({acc}%)\n")

            f.write("\n  By Category:\n")
            for cat, stats in s["by_category"].items():
                acc = round(stats["correct"] / stats["total"] * 100, 1)
                f.write(f"    {cat:<38} {stats['correct']}/{stats['total']} ({acc}%)\n")
            f.write("\n")

        # Per-question responses
        f.write(D + "\n")
        f.write("  DETAILED QUESTION RESPONSES\n")
        f.write(D + "\n\n")

        for model_label, data in all_results.items():
            f.write(f"{'─'*72}\n  {model_label}\n{'─'*72}\n\n")
            for r in data["results"]:
                f.write(f"  [{r['problem_id']}] [{r['difficulty']}] {r['category']}\n")
                f.write(f"  Question        : {r['question']}\n")
                f.write(f"  Correct Answer  : {r['correct_answer']}\n")
                f.write(f"  Trap            : {r['trap_description']}\n")
                f.write(f"  Result          : {'✓ CORRECT' if r['is_correct'] else '✗ WRONG'}\n")
                f.write(f"  Recovery Type   : {r['recovery_type']}\n")
                if r["signals_found"]:
                    sigs = ", ".join(f"'{s['signal']}' @word{s['word_num']}" for s in r["signals_found"][:4])
                    f.write(f"  Signal Words    : {sigs}\n")
                    f.write(f"  Recovery At     : {r['recovery_position_pct']}% through response\n")
                f.write(f"  Latency/Tokens  : {r['latency_ms']}ms | {r['total_tokens']} tokens\n")
                f.write(f"  Response:\n")
                for line in r["model_response"].split("\n"):
                    f.write(f"    {line}\n")
                f.write("\n")

        f.write(D + "\n  END OF REPORT\n" + D + "\n")

    return filename

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("  Reasoning Recovery Benchmark v2 — Harder Traps")
    print("=" * 68)
    print(f"  Models         : {', '.join(MODELS.keys())}")
    print(f"  Problems       : {len(TRAP_PROBLEMS)}")
    print(f"  Difficulty     : Medium → Hard → Very Hard")
    print(f"  Total API calls: {len(TRAP_PROBLEMS) * len(MODELS)}")
    print("=" * 68)

    all_results = run_benchmark()
    summary     = compute_summary(all_results)
    report_file = write_report(all_results, summary)

    print(f"\n{'='*68}")
    print(f"  Report saved → {report_file}")
    print(f"{'='*68}")