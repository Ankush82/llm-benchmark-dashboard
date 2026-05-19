import os
"""
Tool Calling Benchmark — OpenRouter
Tests 3 LLMs on 20 tool-use scenarios across 12 tools.

Evaluation dimensions:
    - Tool selection accuracy  (right tool chosen)
    - Parameter extraction     (correct args passed)
    - Unnecessary tool calls   (false positives)
    - No-tool restraint        (not calling tools when unnecessary)
    - Multi-step chaining      (calling multiple tools correctly)

Usage:
    python3 tool_calling_benchmark.py

Requirements:
    pip3 install openai
"""

import json
import time
from datetime import datetime
from openai import OpenAI

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

MODELS = {
    "Phi-3":     "microsoft/phi-3-medium-128k-instruct",
    "Nemotron":  "nvidia/nemotron-3-super-120b-a12b",
    "Ministral": "mistralai/ministral-14b-2512",
}

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ══════════════════════════════════════════════════════════════════════════════
#  12 TOOLS — JSON Schemas for the API
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather conditions for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location, e.g. 'Paris' or 'New York, NY'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit. Defaults to celsius.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information on a given topic or query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return. Default 5.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a specified recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body text.",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new event on the user's calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Event title or name.",
                    },
                    "date": {
                        "type": "string",
                        "description": "Date of the event in YYYY-MM-DD format.",
                    },
                    "time": {
                        "type": "string",
                        "description": "Start time of the event in HH:MM format (24-hour).",
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Duration of the event in minutes.",
                    },
                },
                "required": ["title", "date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current market price of a stock by its ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. 'AAPL', 'TSLA', 'GOOGL'.",
                    },
                    "currency": {
                        "type": "string",
                        "description": "Currency for the price. Default 'USD'.",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression or perform a calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. '15 * 0.15' or 'sqrt(144)'.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate text from one language to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to translate.",
                    },
                    "target_language": {
                        "type": "string",
                        "description": "Target language, e.g. 'French', 'Spanish', 'Japanese'.",
                    },
                    "source_language": {
                        "type": "string",
                        "description": "Source language. Defaults to 'auto' for auto-detection.",
                    },
                },
                "required": ["text", "target_language"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder for a specific message at a future time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The reminder message or task to remember.",
                    },
                    "datetime": {
                        "type": "string",
                        "description": "When to trigger the reminder, e.g. '2024-12-20 09:00'.",
                    },
                },
                "required": ["message", "datetime"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Fetch the latest news headlines and summaries on a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic or keyword to search news for.",
                    },
                    "num_articles": {
                        "type": "integer",
                        "description": "Number of articles to return. Default 5.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code for news, e.g. 'en', 'fr'. Default 'en'.",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_restaurant",
            "description": "Make a restaurant reservation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the restaurant.",
                    },
                    "date": {
                        "type": "string",
                        "description": "Reservation date in YYYY-MM-DD format.",
                    },
                    "time": {
                        "type": "string",
                        "description": "Reservation time in HH:MM format.",
                    },
                    "party_size": {
                        "type": "integer",
                        "description": "Number of people in the party.",
                    },
                },
                "required": ["name", "date", "time", "party_size"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_directions",
            "description": "Get directions and travel information between two locations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_location": {
                        "type": "string",
                        "description": "Starting location or address.",
                    },
                    "to_location": {
                        "type": "string",
                        "description": "Destination location or address.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["driving", "walking", "transit", "cycling"],
                        "description": "Travel mode. Default 'driving'.",
                    },
                },
                "required": ["from_location", "to_location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_units",
            "description": "Convert a value from one unit of measurement to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "The numeric value to convert.",
                    },
                    "from_unit": {
                        "type": "string",
                        "description": "Source unit, e.g. 'miles', 'kg', 'USD', 'fahrenheit'.",
                    },
                    "to_unit": {
                        "type": "string",
                        "description": "Target unit, e.g. 'kilometers', 'pounds', 'GBP', 'celsius'.",
                    },
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

# ══════════════════════════════════════════════════════════════════════════════
#  20 TEST SCENARIOS
#
#  expected_tools        : tools that MUST be called (all required)
#  expected_params       : key/value substrings to check in tool arguments
#                          (loose match — value just needs to appear somewhere in args)
#  acceptable_tools      : alternative valid tool sets (for ambiguous scenarios)
#  no_tool               : True = model should NOT call any tool
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = [

    # ── EASY: Single tool, obvious ────────────────────────────────────────────
    {
        "id": "S01",
        "category": "Single Tool - Easy",
        "user_message": "What's the weather like in Paris right now?",
        "expected_tools": ["get_weather"],
        "expected_params": {"location": "paris"},
        "acceptable_tools": None,
        "no_tool": False,
        "difficulty": "Easy",
        "notes": "Obvious weather query.",
    },
    {
        "id": "S02",
        "category": "Single Tool - Easy",
        "user_message": "Convert 100 miles to kilometers.",
        "expected_tools": ["convert_units"],
        "expected_params": {"value": "100", "from_unit": "mile", "to_unit": "kilometer"},
        "acceptable_tools": None,
        "no_tool": False,
        "difficulty": "Easy",
        "notes": "Direct unit conversion — value and units must be extracted.",
    },
    {
        "id": "S03",
        "category": "Single Tool - Easy",
        "user_message": "What's Apple's current stock price?",
        "expected_tools": ["get_stock_price"],
        "expected_params": {"ticker": "aapl"},
        "acceptable_tools": None,
        "no_tool": False,
        "difficulty": "Easy",
        "notes": "Model must infer ticker symbol AAPL from company name.",
    },
    {
        "id": "S04",
        "category": "Single Tool - Easy",
        "user_message": "Translate 'Good morning, how are you?' into French.",
        "expected_tools": ["translate_text"],
        "expected_params": {"target_language": "french"},
        "acceptable_tools": None,
        "no_tool": False,
        "difficulty": "Easy",
        "notes": "Basic translation — text and target language extraction.",
    },
    {
        "id": "S05",
        "category": "Single Tool - Easy",
        "user_message": "Calculate 15% of 847.",
        "expected_tools": ["calculate"],
        "expected_params": {"expression": "847"},
        "acceptable_tools": None,
        "no_tool": False,
        "difficulty": "Easy",
        "notes": "Arithmetic — expression must contain 847.",
    },

    # ── MEDIUM: Single tool, parameter extraction ─────────────────────────────
    {
        "id": "S06",
        "category": "Single Tool - Parameter Extraction",
        "user_message": (
            "I'd like to book a table for 4 people at Mario's Italian Kitchen "
            "this Friday at 7:30pm."
        ),
        "expected_tools": ["book_restaurant"],
        "expected_params": {"name": "mario", "party_size": "4", "time": "19:30"},
        "acceptable_tools": None,
        "no_tool": False,
        "difficulty": "Medium",
        "notes": "Multiple param extraction: restaurant name, party size, time.",
    },
    {
        "id": "S07",
        "category": "Single Tool - Parameter Extraction",
        "user_message": "Remind me to call the dentist tomorrow morning at 9am.",
        "expected_tools": ["set_reminder"],
        "expected_params": {"message": "dentist"},
        "acceptable_tools": None,
        "no_tool": False,
        "difficulty": "Medium",
        "notes": "Reminder with message content and time extraction.",
    },
    {
        "id": "S08",
        "category": "Single Tool - Parameter Extraction",
        "user_message": "Show me the latest news about electric vehicles.",
        "expected_tools": ["get_news"],
        "expected_params": {"topic": "electric"},
        "acceptable_tools": None,
        "no_tool": False,
        "difficulty": "Medium",
        "notes": "News query — topic extraction.",
    },
    {
        "id": "S09",
        "category": "Single Tool - Parameter Extraction",
        "user_message": (
            "How do I get from Times Square to JFK Airport using public transport?"
        ),
        "expected_tools": ["get_directions"],
        "expected_params": {"from_location": "times square", "to_location": "jfk", "mode": "transit"},
        "acceptable_tools": None,
        "no_tool": False,
        "difficulty": "Medium",
        "notes": "Directions with mode extraction — 'public transport' → transit.",
    },
    {
        "id": "S10",
        "category": "Single Tool - Parameter Extraction",
        "user_message": (
            "Please send an email to sarah@company.com with subject 'Project Update' "
            "saying the deadline has been moved to next Friday."
        ),
        "expected_tools": ["send_email"],
        "expected_params": {"to": "sarah@company.com", "subject": "project update"},
        "acceptable_tools": None,
        "no_tool": False,
        "difficulty": "Medium",
        "notes": "Email with to/subject/body extraction from natural language.",
    },

    # ── HARD: Multi-tool chaining ─────────────────────────────────────────────
    {
        "id": "S11",
        "category": "Multi-Tool Chain",
        "user_message": (
            "What's the weather in Tokyo right now? "
            "Also, how would I get there from New York?"
        ),
        "expected_tools": ["get_weather", "get_directions"],
        "expected_params": {"location": "tokyo", "from_location": "new york", "to_location": "tokyo"},
        "acceptable_tools": None,
        "no_tool": False,
        "difficulty": "Hard",
        "notes": "Two tools needed: weather + directions.",
    },
    {
        "id": "S12",
        "category": "Multi-Tool Chain",
        "user_message": (
            "What's Tesla's current stock price? "
            "Also convert it to British pounds for me."
        ),
        "expected_tools": ["get_stock_price", "convert_units"],
        "expected_params": {"ticker": "tsla", "to_unit": "gbp"},
        "acceptable_tools": None,
        "no_tool": False,
        "difficulty": "Hard",
        "notes": "Stock price then currency conversion chaining.",
    },
    {
        "id": "S13",
        "category": "Multi-Tool Chain",
        "user_message": (
            "Schedule a meeting called 'Team Standup' for tomorrow at 9am for 30 minutes. "
            "Also set a reminder 10 minutes before it starts."
        ),
        "expected_tools": ["create_calendar_event", "set_reminder"],
        "expected_params": {"title": "standup", "duration_minutes": "30"},
        "acceptable_tools": None,
        "no_tool": False,
        "difficulty": "Hard",
        "notes": "Calendar event + reminder — both must be called.",
    },
    {
        "id": "S14",
        "category": "Multi-Tool Chain",
        "user_message": (
            "Search for information about the Louvre Museum "
            "and get driving directions from London to Paris."
        ),
        "expected_tools": ["search_web", "get_directions"],
        "expected_params": {"query": "louvre", "from_location": "london", "to_location": "paris"},
        "acceptable_tools": None,
        "no_tool": False,
        "difficulty": "Hard",
        "notes": "Web search + directions chaining.",
    },

    # ── NO TOOL NEEDED ────────────────────────────────────────────────────────
    {
        "id": "S15",
        "category": "No Tool Needed",
        "user_message": "What is the capital of France?",
        "expected_tools": [],
        "expected_params": {},
        "acceptable_tools": None,
        "no_tool": True,
        "difficulty": "Easy",
        "notes": "General knowledge — answer directly, no tool needed.",
    },
    {
        "id": "S16",
        "category": "No Tool Needed",
        "user_message": "Write a short haiku about autumn leaves.",
        "expected_tools": [],
        "expected_params": {},
        "acceptable_tools": None,
        "no_tool": True,
        "difficulty": "Easy",
        "notes": "Creative writing — no tool needed.",
    },
    {
        "id": "S17",
        "category": "No Tool Needed",
        "user_message": "Explain the concept of recursion in programming.",
        "expected_tools": [],
        "expected_params": {},
        "acceptable_tools": None,
        "no_tool": True,
        "difficulty": "Medium",
        "notes": "Technical explanation — should not search web, answer from knowledge.",
    },

    # ── TOOL SELECTION TRAPS ──────────────────────────────────────────────────
    {
        "id": "S18",
        "category": "Tool Selection Trap",
        "user_message": "I need to remember to buy groceries.",
        "expected_tools": ["set_reminder"],
        "expected_params": {"message": "groceries"},
        "acceptable_tools": None,
        "no_tool": False,
        "difficulty": "Medium",
        "notes": "Should use set_reminder, NOT create_calendar_event.",
    },
    {
        "id": "S19",
        "category": "Tool Selection Trap",
        "user_message": (
            "I want to know today's temperature in Berlin "
            "in both Celsius and Fahrenheit."
        ),
        "expected_tools": ["get_weather"],
        "expected_params": {"location": "berlin"},
        "acceptable_tools": None,
        "no_tool": False,
        "difficulty": "Hard",
        "notes": "Weather query — trap: might reach for convert_units instead of get_weather.",
    },
    {
        "id": "S20",
        "category": "Tool Selection Trap",
        "user_message": "What is 32 degrees Fahrenheit in Celsius?",
        "expected_tools": ["convert_units"],
        "expected_params": {"value": "32", "from_unit": "fahrenheit", "to_unit": "celsius"},
        "acceptable_tools": [["calculate"], ["convert_units"]],
        "no_tool": False,
        "difficulty": "Medium",
        "notes": "Temperature conversion — convert_units preferred; calculate also acceptable.",
    },
]

# ══════════════════════════════════════════════════════════════════════════════
#  API CALL
# ══════════════════════════════════════════════════════════════════════════════

def call_model_with_tools(model_id, user_message):
    start = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": user_message}],
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
            max_tokens=1024,
        )
        latency = (time.perf_counter() - start) * 1000
        msg = completion.choices[0].message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": args,
                    "raw_arguments": tc.function.arguments,
                })

        return {
            "success":           True,
            "tool_calls":        tool_calls,
            "text_response":     msg.content or "",
            "finish_reason":     completion.choices[0].finish_reason,
            "latency_ms":        round(latency, 1),
            "prompt_tokens":     completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens":      completion.usage.total_tokens,
            "error":             None,
        }
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return {
            "success": False, "tool_calls": [], "text_response": "",
            "finish_reason": "error",
            "latency_ms": round(latency, 1),
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "error": str(e),
        }

# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def args_contain(arguments: dict, key: str, value: str) -> bool:
    """Loose check: does any string representation of the args contain the value?"""
    args_str = json.dumps(arguments).lower()
    return value.lower() in args_str

def evaluate_scenario(scenario: dict, tool_calls: list[dict]) -> dict:
    """
    Score a single model response against a scenario's expected behaviour.

    Returns a dict with:
        score           0-3
        tool_selection  "correct" | "partial" | "wrong" | "unnecessary"
        params_score    0-1  (1 = all key params found)
        details         human-readable breakdown
    """
    called_names  = [tc["name"] for tc in tool_calls]
    expected      = scenario["expected_tools"]
    no_tool       = scenario["no_tool"]
    acceptable    = scenario.get("acceptable_tools")   # list of acceptable tool-name lists

    details = []

    # ── No-tool scenarios ─────────────────────────────────────────────────────
    if no_tool:
        if len(called_names) == 0:
            return {
                "score": 3,
                "tool_selection": "correct",
                "params_score": 1.0,
                "details": ["Correctly answered without calling any tool."],
            }
        else:
            return {
                "score": 0,
                "tool_selection": "unnecessary",
                "params_score": 0.0,
                "details": [f"Should NOT call tools. Called: {called_names}"],
            }

    # ── Check acceptable alternatives ────────────────────────────────────────
    if acceptable:
        for alt_set in acceptable:
            if sorted(called_names) == sorted(alt_set):
                details.append(f"Acceptable alternative tool set used: {alt_set}")
                return {
                    "score": 3,
                    "tool_selection": "correct",
                    "params_score": 1.0,
                    "details": details,
                }

    # ── Tool selection check ──────────────────────────────────────────────────
    missing_tools   = [t for t in expected if t not in called_names]
    extra_tools     = [t for t in called_names if t not in expected]

    if not missing_tools and not extra_tools:
        tool_selection = "correct"
        tool_score     = 2
        details.append(f"Tool selection: correct ({called_names})")
    elif not missing_tools and extra_tools:
        tool_selection = "partial"
        tool_score     = 1
        details.append(f"All required tools called but {len(extra_tools)} extra: {extra_tools}")
    elif missing_tools and not extra_tools:
        tool_selection = "partial"
        tool_score     = 1 if len(missing_tools) < len(expected) else 0
        details.append(f"Missing tools: {missing_tools}")
    else:
        tool_selection = "wrong"
        tool_score     = 0
        details.append(f"Wrong tools. Expected: {expected}. Got: {called_names}")

    if not called_names:
        return {
            "score": 0,
            "tool_selection": "wrong",
            "params_score": 0.0,
            "details": ["No tool called when one was required."],
        }

    # ── Parameter check ───────────────────────────────────────────────────────
    expected_params = scenario.get("expected_params", {})
    if not expected_params:
        params_score = 1.0
        param_hits   = []
    else:
        # Build a merged args dict across all tool calls for loose matching
        all_args = {}
        for tc in tool_calls:
            all_args.update(tc["arguments"])

        hits  = 0
        param_hits = []
        for key, val in expected_params.items():
            found = args_contain(all_args, key, val)
            param_hits.append(f"  param '{key}'='{val}': {'✓' if found else '✗'}")
            if found:
                hits += 1

        params_score = hits / len(expected_params)
        details.extend(param_hits)

    # ── Final score ───────────────────────────────────────────────────────────
    if tool_score == 2 and params_score >= 0.75:
        score = 3
    elif tool_score == 2 and params_score >= 0.4:
        score = 2
    elif tool_score >= 1 or params_score >= 0.5:
        score = 1
    else:
        score = 0

    return {
        "score":          score,
        "tool_selection": tool_selection,
        "params_score":   round(params_score, 2),
        "details":        details,
    }

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

        for scenario in SCENARIOS:
            diff = scenario["difficulty"]
            print(f"\n  [{scenario['id']}] {scenario['category']}  [{diff}]")
            print(f"  Q: {scenario['user_message'][:80]}...")

            result = call_model_with_tools(model_id, scenario["user_message"])

            if result["success"]:
                evaluation = evaluate_scenario(scenario, result["tool_calls"])
            else:
                evaluation = {
                    "score": 0,
                    "tool_selection": "error",
                    "params_score": 0.0,
                    "details": [f"API error: {result['error']}"],
                }

            called_names = [tc["name"] for tc in result["tool_calls"]]
            score_str    = f"{evaluation['score']}/3"

            if evaluation["tool_selection"] == "correct":
                status = "✓"
            elif evaluation["tool_selection"] in ("partial",):
                status = "~"
            else:
                status = "✗"

            print(
                f"  {status} Score: {score_str}  |  Tools called: {called_names or '(none)'}  "
                f"|  {result['latency_ms']}ms  {result['total_tokens']} tok"
            )

            model_results.append({
                "scenario_id":      scenario["id"],
                "category":         scenario["category"],
                "difficulty":       scenario["difficulty"],
                "user_message":     scenario["user_message"],
                "expected_tools":   scenario["expected_tools"],
                "no_tool":          scenario["no_tool"],
                "called_tools":     called_names,
                "tool_calls_raw":   result["tool_calls"],
                "text_response":    result["text_response"],
                "score":            evaluation["score"],
                "tool_selection":   evaluation["tool_selection"],
                "params_score":     evaluation["params_score"],
                "eval_details":     evaluation["details"],
                "latency_ms":       result["latency_ms"],
                "prompt_tokens":    result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens":     result["total_tokens"],
                "error":            result["error"],
            })

            time.sleep(1.5)

        all_results[model_label] = {"model_id": model_id, "results": model_results}

    return all_results

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary(all_results):
    summary = {}

    for model_label, data in all_results.items():
        results = data["results"]
        n       = len(results)

        total_score       = sum(r["score"] for r in results)
        max_score         = n * 3
        pct               = round(total_score / max_score * 100, 1)

        correct_tools     = sum(1 for r in results if r["tool_selection"] == "correct")
        partial_tools     = sum(1 for r in results if r["tool_selection"] == "partial")
        wrong_tools       = sum(1 for r in results if r["tool_selection"] in ("wrong", "error"))
        unnecessary       = sum(1 for r in results if r["tool_selection"] == "unnecessary")

        perfect_scores    = sum(1 for r in results if r["score"] == 3)
        zero_scores       = sum(1 for r in results if r["score"] == 0)

        avg_params        = sum(r["params_score"] for r in results) / n
        avg_latency       = sum(r["latency_ms"]   for r in results) / n
        avg_tokens        = sum(r["total_tokens"]  for r in results) / n

        # By category
        by_category = {}
        for r in results:
            cat = r["category"]
            if cat not in by_category:
                by_category[cat] = {"total_score": 0, "max": 0, "count": 0}
            by_category[cat]["total_score"] += r["score"]
            by_category[cat]["max"]         += 3
            by_category[cat]["count"]       += 1

        cat_pct = {
            cat: round(v["total_score"] / v["max"] * 100, 1)
            for cat, v in by_category.items()
        }

        # By difficulty
        by_difficulty = {}
        for r in results:
            d = r["difficulty"]
            if d not in by_difficulty:
                by_difficulty[d] = {"total_score": 0, "max": 0}
            by_difficulty[d]["total_score"] += r["score"]
            by_difficulty[d]["max"]         += 3

        diff_pct = {
            d: round(v["total_score"] / v["max"] * 100, 1)
            for d, v in by_difficulty.items()
        }

        summary[model_label] = {
            "total_score":     total_score,
            "max_score":       max_score,
            "score_pct":       pct,
            "correct_tools":   correct_tools,
            "partial_tools":   partial_tools,
            "wrong_tools":     wrong_tools,
            "unnecessary":     unnecessary,
            "perfect_scores":  perfect_scores,
            "zero_scores":     zero_scores,
            "avg_params_pct":  round(avg_params * 100, 1),
            "avg_latency_ms":  round(avg_latency, 1),
            "avg_tokens":      round(avg_tokens, 1),
            "by_category":     cat_pct,
            "by_difficulty":   diff_pct,
        }

    return summary

# ══════════════════════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_report(all_results, summary):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"BenchMark_Results/tool_calling_{timestamp}.txt"
    D = "=" * 72
    T = "-" * 72

    categories  = ["Single Tool - Easy", "Single Tool - Parameter Extraction",
                   "Multi-Tool Chain", "No Tool Needed", "Tool Selection Trap"]
    difficulties = ["Easy", "Medium", "Hard"]

    with open(filename, "w", encoding="utf-8") as f:

        # ── Header ────────────────────────────────────────────────────────────
        f.write(D + "\n")
        f.write("  TOOL CALLING BENCHMARK REPORT\n")
        f.write(D + "\n")
        f.write(f"  Date          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Models tested : {', '.join(MODELS.keys())}\n")
        f.write(f"  Tools defined : {len(TOOLS)}\n")
        f.write(f"  Scenarios     : {len(SCENARIOS)}\n")
        f.write(f"  Scoring       : 0-3 per scenario (max {len(SCENARIOS)*3} total)\n")
        f.write(f"  Scoring rubric:\n")
        f.write(f"      3 = Correct tool(s) + correct params\n")
        f.write(f"      2 = Correct tool(s) + minor param gaps\n")
        f.write(f"      1 = Partially correct (missing tool or wrong params)\n")
        f.write(f"      0 = Wrong tool / no tool called / unnecessary call\n")
        f.write(D + "\n\n")

        # ── Overall summary ───────────────────────────────────────────────────
        f.write("OVERALL SCORES\n")
        f.write(T + "\n")
        f.write(
            f"  {'Model':<18} {'Score':>10} {'%':>7} "
            f"{'Perfect':>9} {'Zero':>6} {'Avg Tokens':>12}\n"
        )
        f.write(T + "\n")
        for label, s in summary.items():
            f.write(
                f"  {label:<18} "
                f"  {s['total_score']:>4}/{s['max_score']:<4}"
                f"  {s['score_pct']:>5.1f}%"
                f"  {s['perfect_scores']:>7}/20"
                f"  {s['zero_scores']:>4}/20"
                f"  {s['avg_tokens']:>10.0f}\n"
            )
        f.write(T + "\n\n")

        # ── Tool selection accuracy ───────────────────────────────────────────
        f.write("TOOL SELECTION ACCURACY\n")
        f.write(T + "\n")
        f.write(
            f"  {'Model':<18} {'Correct':>9} {'Partial':>9} "
            f"{'Wrong':>7} {'Unnecessary':>13} {'Param Acc':>11}\n"
        )
        f.write(T + "\n")
        for label, s in summary.items():
            f.write(
                f"  {label:<18}"
                f"  {s['correct_tools']:>7}/20"
                f"  {s['partial_tools']:>7}/20"
                f"  {s['wrong_tools']:>5}/20"
                f"  {s['unnecessary']:>11}/20"
                f"  {s['avg_params_pct']:>9.1f}%\n"
            )
        f.write(T + "\n\n")

        # ── Score by difficulty ───────────────────────────────────────────────
        f.write("SCORES BY DIFFICULTY\n")
        f.write(T + "\n")
        header = f"  {'Difficulty':<14}"
        for label in MODELS:
            header += f"  {label:>12}"
        f.write(header + "\n")
        f.write(T + "\n")
        for diff in difficulties:
            row = f"  {diff:<14}"
            for label, s in summary.items():
                pct = s["by_difficulty"].get(diff, 0)
                row += f"  {pct:>11.1f}%"
            f.write(row + "\n")
        f.write(T + "\n\n")

        # ── Score by category ─────────────────────────────────────────────────
        f.write("SCORES BY CATEGORY\n")
        f.write(T + "\n")
        header = f"  {'Category':<38}"
        for label in MODELS:
            header += f"  {label:>12}"
        f.write(header + "\n")
        f.write(T + "\n")
        for cat in categories:
            row = f"  {cat:<38}"
            scores = {}
            for label, s in summary.items():
                pct = s["by_category"].get(cat, 0)
                scores[label] = pct
                row += f"  {pct:>11.1f}%"
            best = max(scores, key=scores.get) if scores else "-"
            row += f"  ← {best}"
            f.write(row + "\n")
        f.write(T + "\n\n")

        # ── Scenario-by-scenario comparison ──────────────────────────────────
        f.write("SCENARIO-BY-SCENARIO RESULTS\n")
        f.write(T + "\n")
        header = f"  {'ID':<5} {'Category':<38} {'Diff':<8}"
        for label in MODELS:
            header += f"  {label[:8]:>8}"
        f.write(header + "\n")
        f.write(T + "\n")

        for i, scenario in enumerate(SCENARIOS):
            row = f"  {scenario['id']:<5} {scenario['category']:<38} {scenario['difficulty']:<8}"
            for label, data in all_results.items():
                r = data["results"][i]
                marker = "✓" if r["score"] == 3 else ("~" if r["score"] in (1,2) else "✗")
                row += f"  {marker}{r['score']}/3    "
            f.write(row + "\n")
        f.write(T + "\n\n")

        # ── Per-model detail ──────────────────────────────────────────────────
        for model_label, data in all_results.items():
            s = summary[model_label]
            f.write(D + "\n")
            f.write(f"  MODEL: {model_label}   ({MODELS[model_label]})\n")
            f.write(T + "\n")
            f.write(f"  Overall score        : {s['total_score']}/{s['max_score']} ({s['score_pct']}%)\n")
            f.write(f"  Perfect (3/3)        : {s['perfect_scores']}/20\n")
            f.write(f"  Zero (0/3)           : {s['zero_scores']}/20\n")
            f.write(f"  Correct tool sel.    : {s['correct_tools']}/20\n")
            f.write(f"  Partial tool sel.    : {s['partial_tools']}/20\n")
            f.write(f"  Wrong tool sel.      : {s['wrong_tools']}/20\n")
            f.write(f"  Unnecessary calls    : {s['unnecessary']}/20\n")
            f.write(f"  Avg param accuracy   : {s['avg_params_pct']}%\n")
            f.write(f"  Avg latency          : {s['avg_latency_ms']} ms\n")
            f.write(f"  Avg tokens/scenario  : {s['avg_tokens']:.0f}\n\n")

            f.write("  Category breakdown:\n")
            for cat, pct in s["by_category"].items():
                bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                f.write(f"    {cat:<38} [{bar}] {pct:.1f}%\n")
            f.write("\n")

            f.write("  Difficulty breakdown:\n")
            for diff, pct in s["by_difficulty"].items():
                bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                f.write(f"    {diff:<12} [{bar}] {pct:.1f}%\n")
            f.write("\n")

            f.write(T + "\n")
            f.write("  Scenario-by-scenario:\n")
            f.write(T + "\n\n")

            for r in data["results"]:
                marker = "✓ PASS" if r["score"] == 3 else (f"~ {r['score']}/3" if r["score"] > 0 else "✗ FAIL")
                f.write(f"  [{r['scenario_id']}] {r['category']}  [{r['difficulty']}]  {marker}\n")
                f.write(f"  User message    : {r['user_message']}\n")
                f.write(f"  Expected tools  : {r['expected_tools'] or '(none)'}\n")
                f.write(f"  Called tools    : {r['called_tools'] or '(none)'}\n")
                f.write(f"  Tool selection  : {r['tool_selection']}\n")
                f.write(f"  Param accuracy  : {r['params_score']*100:.0f}%\n")
                if r["tool_calls_raw"]:
                    f.write(f"  Tool arguments  :\n")
                    for tc in r["tool_calls_raw"]:
                        f.write(f"    {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)})\n")
                if r["text_response"]:
                    f.write(f"  Text response   : {r['text_response'][:200]}\n")
                f.write(f"  Eval details    :\n")
                for detail in r["eval_details"]:
                    f.write(f"    {detail}\n")
                if r["error"]:
                    f.write(f"  Error           : {r['error']}\n")
                f.write(f"  Latency/Tokens  : {r['latency_ms']}ms | {r['total_tokens']} tokens\n")
                f.write("\n")

        f.write(D + "\n  END OF REPORT\n" + D + "\n")

    return filename

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    total_calls = len(SCENARIOS) * len(MODELS)

    print("=" * 68)
    print("  Tool Calling Benchmark — OpenRouter")
    print("=" * 68)
    print(f"  Models         : {', '.join(MODELS.keys())}")
    print(f"  Tools defined  : {len(TOOLS)}")
    print(f"  Scenarios      : {len(SCENARIOS)}")
    print(f"  Total API calls: {total_calls}")
    print(f"  Categories     : Single Tool Easy/Medium, Multi-Tool,")
    print(f"                   No Tool Needed, Tool Selection Traps")
    print("=" * 68)

    all_results = run_benchmark()
    summary     = compute_summary(all_results)
    report_file = write_report(all_results, summary)

    print(f"\n{'='*68}")
    print("  FINAL SCORES")
    print(f"{'='*68}")
    for label, s in summary.items():
        print(
            f"  {label:<18}: {s['total_score']}/{s['max_score']} ({s['score_pct']}%)  "
            f"Perfect={s['perfect_scores']}/20  "
            f"Correct tool={s['correct_tools']}/20"
        )
    print(f"\n  Report saved → {report_file}")
    print(f"{'='*68}")
