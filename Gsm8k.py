import os

"""
GSM8K Math Benchmark — OpenRouter
Compares multiple LLMs on math word problems.
Outputs results to a .txt report file.
 
Usage:
    python3 gsm8k_benchmark.py
 
Requirements:
    pip3 install openai datasets
"""
 
import re
import time
import random
from datetime import datetime
from openai import OpenAI
from datasets import load_dataset

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
 
MODELS = {
    "Phi-4":            "microsoft/phi-4",
    "Gemma-3":          "nvidia/nemotron-3-super-120b-a12b",
    "Mistral-Nemotron": "mistralai/ministral-14b-2512",
}
 
NUM_QUESTIONS = 10

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def load_gsm8k(n):
    print(f"  Loading {n} questions from GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    samples = random.sample(list(ds), n)
    questions = []
    for s in samples:
        # Expected answer is the number after ####
        expected = s["answer"].split("####")[-1].strip().replace(",", "")
        questions.append({
            "question": s["question"],
            "expected":  expected,
            "full_solution": s["answer"],
        })
    print(f"  Done. Example: {questions[0]['question'][:80]}...\n")
    return questions

def extract_answer(response_text):
    """
    Try several strategies to pull the final numeric answer from the model response.
    GSM8K convention: answer follows ####
    """
    # Strategy 1: model used #### format itself
    hash_match = re.search(r'####\s*\$?(-?\d[\d,]*(?:\.\d+)?)', response_text)
    if hash_match:
        return hash_match.group(1).replace(",", "")
 
    # Strategy 2: "the answer is X" or "= X" at end of response
    answer_match = re.search(
        r'(?:answer is|final answer is|total is|= )\s*\$?(-?\d[\d,]*(?:\.\d+)?)',
        response_text.lower()
    )
    if answer_match:
        return answer_match.group(1).replace(",", "")
 
    # Strategy 3: last number in the response
    numbers = re.findall(r'-?\d[\d,]*(?:\.\d+)?', response_text)
    if numbers:
        return numbers[-1].replace(",", "")
 
    return ""

def call_model(model_id, question):
    # Same Prompt for all models to ensure a fair benchmark
    prompt = (
        f"{question}\n\n"
        "Solve this step by step. "
        "At the end, write your final answer on a new line in this exact format:\n"
        "#### [number only, no units or symbols]"
    )
    start = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,     # deterministic for benchmarking
            max_tokens=512,
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
            "success":            False,
            "response":           "",
            "latency_ms":         round(latency, 1),
            "prompt_tokens":      0,
            "completion_tokens":  0,
            "total_tokens":       0,
            "error":              str(e),
        }

def run_benchmark(questions):
    all_results = {}
 
    for model_label, model_id in MODELS.items():
        print(f"Testing: {model_label}  ({model_id})")
        model_results = []
        correct = 0
 
        for i, q in enumerate(questions):
            print(f"  Q{i+1:02d}/{NUM_QUESTIONS} ...", end=" ", flush=True)
            result = call_model(model_id, q["question"])
 
            if result["success"]:
                extracted = extract_answer(result["response"])
                is_correct = (extracted == q["expected"])
                if is_correct:
                    correct += 1
                status = "✓" if is_correct else "✗"
                print(
                    f"{status}  "
                    f"expected={q['expected']:>8}  "
                    f"got={extracted:>8}  "
                    f"{result['latency_ms']}ms  "
                    f"{result['total_tokens']} tokens"
                )
            
            # If there was an error in the model call
            else:
                is_correct = False
                extracted  = ""
                print(f"ERROR: {result['error']}")
 
            model_results.append({
                "q_num":              i + 1,
                "question":           q["question"],
                "expected":           q["expected"],
                "extracted":          extracted,
                "correct":            is_correct,
                "latency_ms":         result["latency_ms"],
                "prompt_tokens":      result["prompt_tokens"],
                "completion_tokens":  result["completion_tokens"],
                "total_tokens":       result["total_tokens"],
                "response":           result["response"],
                "error":              result["error"],
            })
 
            time.sleep(1.5)   # stay within OpenRouter rate limits
 
        n = len(questions)
        successful = [r for r in model_results if r["total_tokens"] > 0]
 
        all_results[model_label] = {
            "model_id":           model_id,
            "correct":            correct,
            "total":              n,
            "accuracy_pct":       round(correct / n * 100, 1),
            "avg_latency_ms":     round(sum(r["latency_ms"] for r in successful) / len(successful), 1) if successful else 0,
            "avg_prompt_tokens":  round(sum(r["prompt_tokens"] for r in successful) / len(successful), 1) if successful else 0,
            "avg_completion_tokens": round(sum(r["completion_tokens"] for r in successful) / len(successful), 1) if successful else 0,
            "avg_total_tokens":   round(sum(r["total_tokens"] for r in successful) / len(successful), 1) if successful else 0,
            "total_tokens_used":  sum(r["total_tokens"] for r in model_results),
            "questions":          model_results,
        }
 
        print(
            f"\n  → Accuracy: {correct}/{n} ({all_results[model_label]['accuracy_pct']}%)  "
            f"Avg latency: {all_results[model_label]['avg_latency_ms']}ms  "
            f"Avg tokens: {all_results[model_label]['avg_total_tokens']}\n"
        )
 
    return all_results

def write_report(all_results, questions):
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename    = f"gsm8k_results_{timestamp}.txt"
    run_time    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    divider     = "=" * 72
    thin        = "-" * 72
 
    with open(filename, "w", encoding="utf-8") as f:
 
        # ── Header ──────────────────────────────────────────────────────────
        f.write(divider + "\n")
        f.write("  GSM8K MATH WORD PROBLEM BENCHMARK REPORT\n")
        f.write(divider + "\n")
        f.write(f"  Run date       : {run_time}\n")
        f.write(f"  Dataset        : openai/gsm8k  (test split)\n")
        f.write(f"  Questions/model: {NUM_QUESTIONS}\n")
        f.write(f"  Models tested  : {len(MODELS)}\n")
        f.write(f"  Temperature    : 0  (deterministic)\n")
        f.write(divider + "\n\n")
 
        # ── Summary table ────────────────────────────────────────────────────
        f.write("SUMMARY\n")
        f.write(thin + "\n")
        f.write(
            f"  {'Model':<22} {'Accuracy':>10} {'Avg Latency':>13} "
            f"{'Avg Prompt Tok':>16} {'Avg Compl Tok':>15} {'Total Tokens':>14}\n"
        )
        f.write(thin + "\n")
 
        for label, d in all_results.items():
            f.write(
                f"  {label:<22} "
                f"{d['accuracy_pct']:>9.1f}% "
                f"{d['avg_latency_ms']:>11.1f}ms "
                f"{d['avg_prompt_tokens']:>16.1f} "
                f"{d['avg_completion_tokens']:>15.1f} "
                f"{d['total_tokens_used']:>14}\n"
            )
        f.write(thin + "\n\n")
 
        # ── Winner callouts ───────────────────────────────────────────────────
        best_acc = max(all_results.items(), key=lambda x: x[1]["accuracy_pct"])
        best_lat = min(all_results.items(), key=lambda x: x[1]["avg_latency_ms"])
        best_tok = min(all_results.items(), key=lambda x: x[1]["avg_total_tokens"])
 
        f.write("HIGHLIGHTS\n")
        f.write(thin + "\n")
        f.write(f"  Highest accuracy : {best_acc[0]}  ({best_acc[1]['accuracy_pct']}%)\n")
        f.write(f"  Lowest latency   : {best_lat[0]}  ({best_lat[1]['avg_latency_ms']}ms avg)\n")
        f.write(f"  Fewest tokens    : {best_tok[0]}  ({best_tok[1]['avg_total_tokens']} avg tokens)\n")
        f.write(thin + "\n\n")
 
        # ── Per-model detail ──────────────────────────────────────────────────
        for label, d in all_results.items():
            f.write(divider + "\n")
            f.write(f"  MODEL: {label}\n")
            f.write(f"  ID   : {d['model_id']}\n")
            f.write(thin + "\n")
            f.write(f"  Accuracy              : {d['correct']}/{d['total']}  ({d['accuracy_pct']}%)\n")
            f.write(f"  Avg latency           : {d['avg_latency_ms']} ms\n")
            f.write(f"  Avg prompt tokens     : {d['avg_prompt_tokens']}\n")
            f.write(f"  Avg completion tokens : {d['avg_completion_tokens']}\n")
            f.write(f"  Avg total tokens      : {d['avg_total_tokens']}\n")
            f.write(f"  Total tokens used     : {d['total_tokens_used']}\n")
            f.write(thin + "\n\n")
 
            for q in d["questions"]:
                result_marker = "✓ CORRECT" if q["correct"] else "✗ WRONG"
                f.write(f"  Q{q['q_num']:02d}  [{result_marker}]\n")
                f.write(f"  Question : {q['question']}\n")
                f.write(f"  Expected : {q['expected']}\n")
                f.write(f"  Got      : {q['extracted'] if q['extracted'] else '(could not extract)'}\n")
                f.write(f"  Latency  : {q['latency_ms']} ms\n")
                f.write(f"  Tokens   : {q['total_tokens']}  (prompt={q['prompt_tokens']}, completion={q['completion_tokens']})\n")
                f.write(f"  Response :\n")
                # Wrap response at 68 chars for readability
                response_lines = q["response"].split("\n") if q["response"] else ["(no response)"]
                for line in response_lines:
                    f.write(f"    {line}\n")
                if q["error"]:
                    f.write(f"  Error    : {q['error']}\n")
                f.write("\n")
 
            f.write("\n")
 
        f.write(divider + "\n")
        f.write("  END OF REPORT\n")
        f.write(divider + "\n")
 
    return filename
 
# ── Main ───────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("=" * 72)
    print("  GSM8K Math Benchmark — OpenRouter")
    print("=" * 72)
    print(f"  Models    : {', '.join(MODELS.keys())}")
    print(f"  Questions : {NUM_QUESTIONS} per model")
    print(f"  Total API calls: {NUM_QUESTIONS * len(MODELS)}")
    print("=" * 72 + "\n")
 
    questions   = load_gsm8k(NUM_QUESTIONS)
    all_results = run_benchmark(questions)
    report_file = write_report(all_results, questions)
 
    print("=" * 72)
    print(f"  Report saved → {report_file}")
    print("=" * 72)