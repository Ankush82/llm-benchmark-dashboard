import os
"""
Code Writing Benchmark — OpenRouter
Tests LLMs on 10 code writing problems across difficulty levels.

Modes:
    Standard  : python3 code_writing_benchmark.py
    Consistency: python3 code_writing_benchmark.py --consistency 3
    Thinking  : python3 code_writing_benchmark.py --thinking

Standard mode  — single run, temperature=0, deterministic.
Consistency    — N runs per problem at temperature=0.7, reports score variance.
Thinking       — Phi-4-Reasoning-Plus + Nemotron-Reasoning with chain-of-thought.

Requirements:
    pip3 install openai
"""

import re
import time
import json
import argparse
from datetime import datetime
from openai import OpenAI

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

STANDARD_MODELS = {
    "Phi-4":     "microsoft/phi-4",
    "Nemotron":  "nvidia/nemotron-3-super-120b-a12b",
    "Ministral": "mistralai/ministral-14b-2512",
}

THINKING_MODELS = {
    "Phi-4-Reasoning":    "microsoft/phi-4-reasoning-plus:free",
    "Nemotron-Reasoning": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
}

JUDGE_MODEL = "google/gemma-3-27b-it"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ══════════════════════════════════════════════════════════════════════════════
#  10 CODE WRITING PROBLEMS
# ══════════════════════════════════════════════════════════════════════════════

CODE_PROBLEMS = [

    # ── EASY ──────────────────────────────────────────────────────────────────
    {
        "id":         "code_01",
        "category":   "Algorithm - Basic",
        "difficulty": "Easy",
        "prompt": (
            "Write a Python function `fizzbuzz(n)` that returns a list of strings "
            "for numbers 1 through n:\n"
            "  - 'FizzBuzz' if divisible by both 3 and 5\n"
            "  - 'Fizz' if divisible by 3 only\n"
            "  - 'Buzz' if divisible by 5 only\n"
            "  - The number as a string otherwise\n\n"
            "Include a docstring and at least 2 example assertions."
        ),
        "judge_criteria": (
            "Correct output for all cases (multiples of 15, 3, 5, and other). "
            "Proper docstring. At least 2 working assertions. Clean code with no unnecessary complexity."
        ),
    },
    {
        "id":         "code_02",
        "category":   "Algorithm - Search",
        "difficulty": "Easy",
        "prompt": (
            "Implement a Python function `binary_search(arr, target)` that:\n"
            "  - Takes a sorted list and a target value\n"
            "  - Returns the index of the target if found\n"
            "  - Returns -1 if not found\n"
            "  - Must be O(log n) — no linear scan\n\n"
            "Include a docstring, time complexity note, and 3 test cases covering "
            "found, not found, and empty list."
        ),
        "judge_criteria": (
            "Correct binary search logic (not linear scan). O(log n) implementation. "
            "Handles empty list. All 3 test cases included and correct."
        ),
    },

    # ── MEDIUM ─────────────────────────────────────────────────────────────────
    {
        "id":         "code_03",
        "category":   "Algorithm - Hash Map",
        "difficulty": "Medium",
        "prompt": (
            "Write a Python function `two_sum(nums, target)` that:\n"
            "  - Given a list of integers and a target, returns the indices of the "
            "two numbers that add up to the target\n"
            "  - Each input has exactly one solution\n"
            "  - Must use O(n) time complexity (hash map approach, not brute force)\n\n"
            "Show the O(n) approach clearly. Include docstring and 3 test cases."
        ),
        "judge_criteria": (
            "Must use a hash map (dict) approach — O(n), not O(n²) brute force. "
            "Correct indices returned. Docstring present. 3 test cases correct."
        ),
    },
    {
        "id":         "code_04",
        "category":   "Data Structures - Stack",
        "difficulty": "Medium",
        "prompt": (
            "Write a Python function `is_valid_brackets(s)` that:\n"
            "  - Takes a string containing only '(', ')', '{', '}', '[', ']'\n"
            "  - Returns True if the brackets are properly matched and nested\n"
            "  - Returns False otherwise\n"
            "  - Must use a stack\n\n"
            "Examples: '()[]{}' → True, '([)]' → False, '{[]}' → True, '' → True\n"
            "Include docstring and 5 test cases."
        ),
        "judge_criteria": (
            "Stack-based implementation (not regex). Correct for all bracket types. "
            "Handles empty string. All 5 test cases present and correct."
        ),
    },
    {
        "id":         "code_05",
        "category":   "Recursion - Dynamic Programming",
        "difficulty": "Medium",
        "prompt": (
            "Write a Python function `fib(n)` using memoization (not iterative, not naive recursion) that:\n"
            "  - Returns the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1)\n"
            "  - Uses functools.lru_cache or a manual memo dict\n"
            "  - Includes a comment explaining why memoization reduces from O(2^n) to O(n)\n\n"
            "Also write a second version `fib_iterative(n)` for comparison.\n"
            "Show both in action with fib(10) and fib(30)."
        ),
        "judge_criteria": (
            "Memoization correctly implemented (not naive recursion). Time complexity comment present and accurate. "
            "Iterative version also included. Both produce correct outputs for fib(10)=55 and fib(30)=832040."
        ),
    },
    {
        "id":         "code_06",
        "category":   "Recursion - Tree",
        "difficulty": "Medium",
        "prompt": (
            "Write a Python function `flatten(nested)` that:\n"
            "  - Takes an arbitrarily nested list, e.g. [1, [2, [3, [4]], 5], 6]\n"
            "  - Returns a flat list: [1, 2, 3, 4, 5, 6]\n"
            "  - Works for any depth of nesting\n"
            "  - Handles empty lists and mixed types (ints, strings, nested lists)\n\n"
            "Write both a recursive version and a generator-based version.\n"
            "Include 3 test cases."
        ),
        "judge_criteria": (
            "Both recursive and generator versions present and correct. "
            "Handles arbitrary depth. Handles mixed types and empty lists. 3 correct test cases."
        ),
    },

    # ── HARD ───────────────────────────────────────────────────────────────────
    {
        "id":         "code_07",
        "category":   "Data Structures - LRU Cache",
        "difficulty": "Hard",
        "prompt": (
            "Implement an LRU (Least Recently Used) Cache class in Python:\n\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity: int)\n"
            "    def get(self, key: int) -> int       # return -1 if not found\n"
            "    def put(self, key: int, value: int)  # evict LRU if at capacity\n\n"
            "Both get() and put() must be O(1).\n"
            "Do NOT use OrderedDict — implement using a doubly linked list + hash map.\n"
            "Include a usage example demonstrating eviction."
        ),
        "judge_criteria": (
            "O(1) get and put — must use doubly linked list + hash map, not OrderedDict. "
            "Correct eviction of least recently used item. Usage example demonstrates eviction correctly."
        ),
    },
    {
        "id":         "code_08",
        "category":   "Algorithm - Intervals",
        "difficulty": "Hard",
        "prompt": (
            "Write a Python function `merge_intervals(intervals)` that:\n"
            "  - Takes a list of [start, end] intervals\n"
            "  - Merges all overlapping intervals\n"
            "  - Returns the merged intervals sorted by start time\n\n"
            "Example: [[1,3],[2,6],[8,10],[15,18]] → [[1,6],[8,10],[15,18]]\n\n"
            "Include:\n"
            "  - Clear explanation of the approach in comments\n"
            "  - Time complexity analysis\n"
            "  - 4 test cases: normal overlap, no overlap, all overlapping, single interval"
        ),
        "judge_criteria": (
            "Correct merge logic (sort then sweep). Handles all 4 test cases. "
            "Time complexity O(n log n) noted. Comments explain the approach clearly."
        ),
    },
    {
        "id":         "code_09",
        "category":   "Design Patterns - Decorator",
        "difficulty": "Hard",
        "prompt": (
            "Write a Python decorator factory `@retry(max_attempts=3, delay=1.0, exceptions=(Exception,))` that:\n"
            "  - Retries the decorated function up to max_attempts times\n"
            "  - Waits delay seconds between retries\n"
            "  - Only catches the specified exception types\n"
            "  - Logs each attempt with the attempt number and the error message\n"
            "  - Raises the last exception if all attempts are exhausted\n"
            "  - Preserves the original function's __name__ and __doc__ using functools.wraps\n\n"
            "Show a usage example with a function that fails twice then succeeds."
        ),
        "judge_criteria": (
            "Factory decorator (not simple decorator). max_attempts, delay, exceptions all configurable. "
            "functools.wraps used. Logs each attempt. Raises on exhaustion. Usage example shows 2 failures then success."
        ),
    },
    {
        "id":         "code_10",
        "category":   "Debugging",
        "difficulty": "Hard",
        "prompt": (
            "The following Python function has multiple bugs. Find and fix ALL of them, "
            "then explain each fix:\n\n"
            "```python\n"
            "def find_duplicates(lst):\n"
            "    \"\"\"Return a list of all duplicate values in lst.\"\"\"\n"
            "    seen = {}\n"
            "    duplicates = []\n"
            "    for item in lst:\n"
            "        if item in seen:\n"
            "            duplicates.append(item)\n"
            "        else:\n"
            "            seen[item] = True\n"
            "    return list(set(duplicates))  # Bug: loses order and adds extra dupes\n"
            "\n"
            "# Test\n"
            "print(find_duplicates([1, 2, 3, 2, 4, 3, 3]))  # Should be [2, 3], NOT [2, 3, 3]\n"
            "print(find_duplicates([]))                      # Should be []\n"
            "print(find_duplicates([5, 5, 5, 5]))            # Should be [5], NOT [5, 5, 5]\n"
            "```\n\n"
            "The fixed function must: return each duplicate exactly once, preserve insertion order, "
            "and pass all three test cases."
        ),
        "judge_criteria": (
            "All bugs identified and fixed. Fixed function returns each duplicate exactly once. "
            "Preserves insertion order. All 3 test cases pass. Each fix clearly explained."
        ),
    },
]

# ══════════════════════════════════════════════════════════════════════════════
#  LLM JUDGE
# ══════════════════════════════════════════════════════════════════════════════

def build_judge_prompt(problem, response):
    return f"""You are an expert Python code reviewer evaluating an AI assistant's response to a coding problem.

Score the response 1-10 using this rubric:
1-3  : Wrong approach, incorrect code, major errors
4-5  : Partially correct, significant gaps or bugs
6-7  : Mostly correct, minor issues (edge cases, style)
8-9  : Fully correct, clean code, meets all requirements
10   : Perfect — correct, clean, well-documented, best practice

Problem given to the model:
{problem["prompt"]}

Specific evaluation criteria:
{problem["judge_criteria"]}

Model's response:
{response}

Output ONLY in this exact format:
SCORE: [number 1-10]
REASON: [2-3 sentence justification referencing the criteria above]"""


def call_judge(problem, response, max_retries=2):
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": build_judge_prompt(problem, response)}],
                temperature=0,
                max_tokens=200,
            )
            raw = completion.choices[0].message.content.strip()
            score_match  = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', raw)
            reason_match = re.search(r'REASON:\s*(.+)', raw, re.DOTALL)
            score  = float(score_match.group(1))  if score_match  else -1
            reason = reason_match.group(1).strip() if reason_match else raw
            score  = max(1.0, min(10.0, score))
            return {"score": score, "reason": reason, "success": True}
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2.0)
    return {"score": -1, "reason": "Judge failed", "success": False}

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL CALL
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an expert Python developer. Write clean, correct, well-documented code. "
    "Follow the requirements exactly. Include all requested test cases."
)

def call_model(model_id, prompt, temperature=0, thinking=False):
    start = time.perf_counter()
    try:
        kwargs = dict(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=temperature,
            max_tokens=2000,
        )
        if thinking:
            kwargs["extra_body"] = {"reasoning": {"enabled": True}}

        completion = client.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - start) * 1000
        msg = completion.choices[0].message

        # Capture reasoning content if available
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""

        return {
            "success":            True,
            "response":           msg.content or "",
            "reasoning":          reasoning,
            "latency_ms":         round(latency, 1),
            "prompt_tokens":      completion.usage.prompt_tokens,
            "completion_tokens":  completion.usage.completion_tokens,
            "total_tokens":       completion.usage.total_tokens,
            "error":              None,
        }
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return {
            "success": False, "response": "", "reasoning": "",
            "latency_ms": round(latency, 1),
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "error": str(e),
        }

# ══════════════════════════════════════════════════════════════════════════════
#  STANDARD RUN
# ══════════════════════════════════════════════════════════════════════════════

def run_standard(models, thinking=False):
    all_results = {}
    temperature = 0

    for model_label, model_id in models.items():
        print(f"\n{'═'*68}")
        print(f"  Model: {model_label}  ({model_id})")
        if thinking:
            print(f"  Mode : THINKING ENABLED")
        print(f"{'═'*68}")

        model_results = []

        for prob in CODE_PROBLEMS:
            print(f"\n  [{prob['id']}] {prob['category']}  [{prob['difficulty']}]")
            print(f"  ...", end=" ", flush=True)

            result = call_model(model_id, prob["prompt"], temperature=temperature, thinking=thinking)

            if result["success"]:
                time.sleep(0.5)
                judge = call_judge(prob, result["response"])
                score = judge["score"]
                print(f"Score: {score}/10  ({result['latency_ms']}ms, {result['total_tokens']} tok)")
                if thinking and result["reasoning"]:
                    reasoning_words = len(result["reasoning"].split())
                    print(f"  Reasoning: {reasoning_words} words")
            else:
                judge = {"score": -1, "reason": f"API error: {result['error']}", "success": False}
                score = -1
                print(f"ERROR: {result['error']}")

            model_results.append({
                "problem_id":    prob["id"],
                "category":      prob["category"],
                "difficulty":    prob["difficulty"],
                "score":         score,
                "judge_reason":  judge["reason"],
                "response":      result["response"],
                "reasoning":     result["reasoning"],
                "latency_ms":    result["latency_ms"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens":  result["total_tokens"],
                "error":         result["error"],
            })

            time.sleep(1.5)

        all_results[model_label] = {"model_id": model_id, "results": model_results}

    return all_results

# ══════════════════════════════════════════════════════════════════════════════
#  CONSISTENCY RUN  (N runs per problem)
# ══════════════════════════════════════════════════════════════════════════════

def run_consistency(models, n_runs):
    all_results = {}

    for model_label, model_id in models.items():
        print(f"\n{'═'*68}")
        print(f"  Model: {model_label}  ({model_id})  [{n_runs} runs/problem]")
        print(f"{'═'*68}")

        model_results = []

        for prob in CODE_PROBLEMS:
            print(f"\n  [{prob['id']}] {prob['category']}  [{prob['difficulty']}]")
            run_scores   = []
            run_details  = []

            for run_num in range(1, n_runs + 1):
                print(f"    Run {run_num}/{n_runs} ...", end=" ", flush=True)

                result = call_model(model_id, prob["prompt"], temperature=0.7)

                if result["success"]:
                    time.sleep(0.5)
                    judge = call_judge(prob, result["response"])
                    score = judge["score"]
                    print(f"{score}/10  ({result['latency_ms']}ms)")
                else:
                    judge = {"score": -1, "reason": f"Error: {result['error']}", "success": False}
                    score = -1
                    print(f"ERROR")

                run_scores.append(score)
                run_details.append({
                    "run":           run_num,
                    "score":         score,
                    "judge_reason":  judge["reason"],
                    "response":      result["response"],
                    "latency_ms":    result["latency_ms"],
                    "total_tokens":  result["total_tokens"],
                    "error":         result["error"],
                })

                time.sleep(1.5)

            valid_scores = [s for s in run_scores if s > 0]
            if len(valid_scores) >= 2:
                mean   = sum(valid_scores) / len(valid_scores)
                var    = sum((s - mean) ** 2 for s in valid_scores) / len(valid_scores)
                std    = var ** 0.5
                spread = max(valid_scores) - min(valid_scores)
                # Consistency: 100% if std=0, drops as variance grows
                consistency_pct = max(0, round(100 - (std / 10 * 100), 1))
            elif len(valid_scores) == 1:
                mean = valid_scores[0]
                std  = 0.0
                spread = 0.0
                consistency_pct = 100.0
            else:
                mean = 0.0
                std  = 0.0
                spread = 0.0
                consistency_pct = 0.0

            print(
                f"    → Mean: {round(mean,2)}/10  StdDev: {round(std,2)}  "
                f"Spread: {round(spread,1)}  Consistency: {consistency_pct}%"
            )

            model_results.append({
                "problem_id":       prob["id"],
                "category":         prob["category"],
                "difficulty":       prob["difficulty"],
                "runs":             run_details,
                "scores":           run_scores,
                "mean_score":       round(mean, 2),
                "std_dev":          round(std, 2),
                "min_score":        min(valid_scores) if valid_scores else 0,
                "max_score":        max(valid_scores) if valid_scores else 0,
                "spread":           round(spread, 1),
                "consistency_pct":  consistency_pct,
            })

        all_results[model_label] = {"model_id": model_id, "results": model_results}

    return all_results

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary_standard(all_results):
    summary = {}
    for model_label, data in all_results.items():
        results = data["results"]
        valid   = [r for r in results if r["score"] > 0]
        n       = len(results)

        avg_score   = sum(r["score"]       for r in valid) / len(valid) if valid else 0
        avg_latency = sum(r["latency_ms"]  for r in results) / n
        avg_tokens  = sum(r["total_tokens"] for r in results) / n

        by_diff = {}
        for r in valid:
            d = r["difficulty"]
            if d not in by_diff:
                by_diff[d] = []
            by_diff[d].append(r["score"])

        diff_avgs = {d: round(sum(s)/len(s), 2) for d, s in by_diff.items()}

        summary[model_label] = {
            "avg_score":      round(avg_score, 2),
            "avg_latency_ms": round(avg_latency, 1),
            "avg_tokens":     round(avg_tokens, 1),
            "by_difficulty":  diff_avgs,
        }
    return summary


def compute_summary_consistency(all_results):
    summary = {}
    for model_label, data in all_results.items():
        results = data["results"]
        n       = len(results)

        avg_mean         = sum(r["mean_score"]      for r in results) / n
        avg_std          = sum(r["std_dev"]          for r in results) / n
        avg_consistency  = sum(r["consistency_pct"] for r in results) / n
        most_consistent  = min(results, key=lambda r: r["std_dev"])
        least_consistent = max(results, key=lambda r: r["std_dev"])

        summary[model_label] = {
            "avg_mean_score":    round(avg_mean, 2),
            "avg_std_dev":       round(avg_std, 2),
            "avg_consistency":   round(avg_consistency, 1),
            "most_consistent":   most_consistent["problem_id"],
            "least_consistent":  least_consistent["problem_id"],
        }
    return summary

# ══════════════════════════════════════════════════════════════════════════════
#  REPORT WRITERS
# ══════════════════════════════════════════════════════════════════════════════

def write_standard_report(all_results, summary, filename_prefix, thinking=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"BenchMark_Results/{filename_prefix}_{timestamp}.txt"
    D = "=" * 72
    T = "-" * 72
    difficulties = ["Easy", "Medium", "Hard"]

    with open(filename, "w", encoding="utf-8") as f:

        f.write(D + "\n")
        mode_label = "THINKING MODE" if thinking else "STANDARD"
        f.write(f"  CODE WRITING BENCHMARK — {mode_label}\n")
        f.write(D + "\n")
        f.write(f"  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Models     : {', '.join(all_results.keys())}\n")
        f.write(f"  Problems   : {len(CODE_PROBLEMS)}\n")
        f.write(f"  Judge      : {JUDGE_MODEL}\n")
        f.write(f"  Scoring    : 1-10 per problem (LLM judge)\n")
        if thinking:
            f.write(f"  Thinking   : enabled (chain-of-thought reasoning)\n")
        f.write(D + "\n\n")

        # Overall scores
        f.write("OVERALL SCORES\n")
        f.write(T + "\n")
        f.write(f"  {'Model':<24} {'Avg Score':>10} {'Avg Latency':>13} {'Avg Tokens':>12}\n")
        f.write(T + "\n")
        for label, s in summary.items():
            f.write(
                f"  {label:<24}"
                f"  {s['avg_score']:>8.2f}/10"
                f"  {s['avg_latency_ms']:>11.1f}ms"
                f"  {s['avg_tokens']:>10.0f}\n"
            )
        f.write(T + "\n\n")

        # By difficulty
        f.write("SCORES BY DIFFICULTY\n")
        f.write(T + "\n")
        header = f"  {'Difficulty':<12}"
        for label in all_results:
            header += f"  {label:>14}"
        f.write(header + "\n")
        f.write(T + "\n")
        for diff in difficulties:
            row = f"  {diff:<12}"
            for label, s in summary.items():
                score = s["by_difficulty"].get(diff, 0)
                row += f"  {score:>13.2f}/10"
            f.write(row + "\n")
        f.write(T + "\n\n")

        # Problem-by-problem grid
        f.write("PROBLEM-BY-PROBLEM SCORES\n")
        f.write(T + "\n")
        header = f"  {'ID':<10} {'Category':<30} {'Diff':<8}"
        for label in all_results:
            header += f"  {label[:10]:>10}"
        f.write(header + "\n")
        f.write(T + "\n")
        for i, prob in enumerate(CODE_PROBLEMS):
            row = f"  {prob['id']:<10} {prob['category']:<30} {prob['difficulty']:<8}"
            for label, data in all_results.items():
                r = data["results"][i]
                s = f"{r['score']:.1f}" if r["score"] > 0 else "ERR"
                row += f"  {s:>10}"
            f.write(row + "\n")
        f.write(T + "\n\n")

        # Per-model detail
        for model_label, data in all_results.items():
            s = summary[model_label]
            f.write(D + "\n")
            f.write(f"  MODEL: {model_label}   ({data['model_id']})\n")
            f.write(T + "\n")
            f.write(f"  Avg score    : {s['avg_score']}/10\n")
            f.write(f"  Avg latency  : {s['avg_latency_ms']} ms\n")
            f.write(f"  Avg tokens   : {s['avg_tokens']:.0f}\n\n")

            f.write("  By difficulty:\n")
            for diff in difficulties:
                score = s["by_difficulty"].get(diff, "N/A")
                bar = "█" * int(float(score)) + "░" * (10 - int(float(score))) if score != "N/A" else "░" * 10
                f.write(f"    {diff:<8} [{bar}] {score}/10\n")
            f.write("\n")

            f.write(T + "\n")
            for r in data["results"]:
                f.write(f"\n  [{r['problem_id']}] {r['category']}  [{r['difficulty']}]\n")
                f.write(f"  Score        : {r['score']}/10\n")
                f.write(f"  Judge reason : {r['judge_reason']}\n")
                f.write(f"  Latency/Tok  : {r['latency_ms']}ms | {r['total_tokens']} tokens\n")
                if thinking and r["reasoning"]:
                    words = len(r["reasoning"].split())
                    f.write(f"  Reasoning    : {words} words\n")
                    f.write(f"  Reasoning excerpt:\n")
                    snippet = r["reasoning"][:500] + ("..." if len(r["reasoning"]) > 500 else "")
                    for line in snippet.split("\n"):
                        f.write(f"    {line}\n")
                f.write(f"  Response:\n")
                for line in r["response"].split("\n"):
                    f.write(f"    {line}\n")
                if r["error"]:
                    f.write(f"  Error        : {r['error']}\n")

        f.write("\n" + D + "\n  END OF REPORT\n" + D + "\n")

    return filename


def write_consistency_report(all_results, summary):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"BenchMark_Results/code_consistency_{timestamp}.txt"
    D = "=" * 72
    T = "-" * 72

    with open(filename, "w", encoding="utf-8") as f:

        f.write(D + "\n")
        f.write("  CODE WRITING BENCHMARK — CONSISTENCY ANALYSIS\n")
        f.write(D + "\n")
        f.write(f"  Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Models   : {', '.join(all_results.keys())}\n")
        f.write(f"  Problems : {len(CODE_PROBLEMS)}\n")
        f.write(f"  Judge    : {JUDGE_MODEL}\n")
        f.write(f"  Metric   : StdDev of scores across runs (lower = more consistent)\n")
        f.write(D + "\n\n")

        # Summary
        f.write("CONSISTENCY SUMMARY\n")
        f.write(T + "\n")
        f.write(f"  {'Model':<24} {'Avg Score':>10} {'Avg StdDev':>12} {'Consistency':>13}\n")
        f.write(T + "\n")
        for label, s in summary.items():
            bar = "█" * int(s["avg_consistency"] / 10) + "░" * (10 - int(s["avg_consistency"] / 10))
            f.write(
                f"  {label:<24}"
                f"  {s['avg_mean_score']:>8.2f}/10"
                f"  {s['avg_std_dev']:>10.2f}"
                f"  {s['avg_consistency']:>10.1f}%  [{bar}]\n"
            )
        f.write(T + "\n\n")

        # Problem-by-problem consistency grid
        f.write("CONSISTENCY BY PROBLEM  (StdDev — lower is better)\n")
        f.write(T + "\n")
        header = f"  {'ID':<10} {'Category':<30} {'Diff':<8}"
        for label in all_results:
            header += f"  {label[:10]:>14}"
        f.write(header + "\n")
        f.write(T + "\n")
        for i, prob in enumerate(CODE_PROBLEMS):
            row = f"  {prob['id']:<10} {prob['category']:<30} {prob['difficulty']:<8}"
            for label, data in all_results.items():
                r = data["results"][i]
                row += f"  {r['mean_score']:>5.1f}±{r['std_dev']:<5.2f}  "
            f.write(row + "\n")
        f.write(T + "\n\n")

        # Per-model detail
        for model_label, data in all_results.items():
            s = summary[model_label]
            f.write(D + "\n")
            f.write(f"  MODEL: {model_label}   ({data['model_id']})\n")
            f.write(T + "\n")
            f.write(f"  Avg mean score      : {s['avg_mean_score']}/10\n")
            f.write(f"  Avg std deviation   : {s['avg_std_dev']}\n")
            f.write(f"  Avg consistency     : {s['avg_consistency']}%\n")
            f.write(f"  Most consistent     : {s['most_consistent']}\n")
            f.write(f"  Least consistent    : {s['least_consistent']}\n\n")

            for r in data["results"]:
                f.write(f"  [{r['problem_id']}] {r['category']}  [{r['difficulty']}]\n")
                f.write(f"  Mean={r['mean_score']}/10  StdDev={r['std_dev']}  "
                        f"Min={r['min_score']}  Max={r['max_score']}  "
                        f"Spread={r['spread']}  Consistency={r['consistency_pct']}%\n")
                f.write(f"  Scores by run: {r['scores']}\n")
                for run in r["runs"]:
                    f.write(f"    Run {run['run']}: {run['score']}/10  {run['latency_ms']}ms\n")
                    f.write(f"    Judge: {run['judge_reason']}\n")
                f.write("\n")

        f.write(D + "\n  END OF REPORT\n" + D + "\n")

    return filename

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Writing Benchmark")
    parser.add_argument(
        "--consistency", type=int, default=0, metavar="N",
        help="Run each problem N times to measure score consistency (uses temperature=0.7)",
    )
    parser.add_argument(
        "--thinking", action="store_true",
        help="Run thinking mode: Phi-4-Reasoning-Plus + Nemotron-Reasoning",
    )
    args = parser.parse_args()

    print("=" * 68)
    print("  Code Writing Benchmark — OpenRouter")
    print("=" * 68)
    print(f"  Problems   : {len(CODE_PROBLEMS)}")
    print(f"  Judge      : {JUDGE_MODEL}")

    if args.thinking:
        models = THINKING_MODELS
        mode   = "thinking"
        total_calls = len(CODE_PROBLEMS) * len(models) * 2  # model + judge
        print(f"  Mode       : THINKING  (Phi-4-Reasoning + Nemotron-Reasoning)")
        print(f"  Models     : {', '.join(models.keys())}")
        print(f"  Total calls: ~{total_calls}")
        print("=" * 68)

        all_results = run_standard(models, thinking=True)
        summary     = compute_summary_standard(all_results)
        report_file = write_standard_report(all_results, summary, "code_thinking", thinking=True)

    elif args.consistency > 0:
        n_runs = args.consistency
        models = STANDARD_MODELS
        total_calls = len(CODE_PROBLEMS) * len(models) * n_runs * 2
        print(f"  Mode       : CONSISTENCY  ({n_runs} runs/problem)")
        print(f"  Models     : {', '.join(models.keys())}")
        print(f"  Temperature: 0.7 (variation across runs)")
        print(f"  Total calls: ~{total_calls}")
        print("=" * 68)

        all_results = run_consistency(models, n_runs)
        summary     = compute_summary_consistency(all_results)
        report_file = write_consistency_report(all_results, summary)

    else:
        models = STANDARD_MODELS
        total_calls = len(CODE_PROBLEMS) * len(models) * 2
        print(f"  Mode       : STANDARD  (single run, temperature=0)")
        print(f"  Models     : {', '.join(models.keys())}")
        print(f"  Total calls: ~{total_calls}")
        print("=" * 68)

        all_results = run_standard(models)
        summary     = compute_summary_standard(all_results)
        report_file = write_standard_report(all_results, summary, "code_writing")

    print(f"\n{'='*68}")
    print("  FINAL SCORES")
    print(f"{'='*68}")

    if args.consistency > 0:
        for label, s in summary.items():
            print(
                f"  {label:<24}: mean={s['avg_mean_score']}/10  "
                f"stddev={s['avg_std_dev']}  consistency={s['avg_consistency']}%"
            )
    else:
        for label, s in summary.items():
            print(f"  {label:<24}: {s['avg_score']}/10  avg_latency={s['avg_latency_ms']}ms")

    print(f"\n  Report saved → {report_file}")
    print(f"{'='*68}")
