"""
LLM Benchmark Dashboard
Run benchmarks and visualize results from one place.

Usage:
    streamlit run dashboard.py
    python3 -m streamlit run dashboard.py
"""

import re
import os
import sys
import time
import queue as queue_module
import subprocess
import threading
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BENCH_DIR    = Path(__file__).parent
RESULTS_DIR  = BENCH_DIR / "BenchMark_Results"
RESULTS_DIR.mkdir(exist_ok=True)

# API key — from Streamlit secrets (cloud) or local secrets.toml
try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    OPENROUTER_API_KEY = ""

BENCHMARKS = {
    "GSM8K Math":               {"script": "Gsm8k.py",                    "modes": ["standard"]},
    "MT-Bench Chat":             {"script": "mt_bench.py",                 "modes": ["standard"]},
    "Reasoning Recovery v2":    {"script": "Recovery.py",                  "modes": ["standard"]},
    "Tool Calling":             {"script": "tool_calling_benchmark.py",    "modes": ["standard"]},
    "Code Writing — Standard":  {"script": "code_writing_benchmark.py",   "modes": ["standard"]},
    "Code Writing — Thinking":  {"script": "code_writing_benchmark.py",   "modes": ["thinking"]},
    "Code Writing — Consistency":{"script": "code_writing_benchmark.py",  "modes": ["consistency"]},
}

MODEL_COLORS = {
    "Phi-4":              "#4C9BE8",
    "Phi-4-Reasoning":    "#1A5FA8",
    "Nemotron":           "#76B900",
    "Nemotron-Reasoning": "#4A7A00",
    "Ministral":          "#FF6B35",
}

def model_color(name):
    for key, color in MODEL_COLORS.items():
        if key.lower() in name.lower():
            return color
    return "#888888"

# ══════════════════════════════════════════════════════════════════════════════
#  RESULT FILE PARSERS
# ══════════════════════════════════════════════════════════════════════════════

def detect_benchmark_type(filename):
    name = filename.lower()
    if "gsm8k"          in name: return "gsm8k"
    if "mt_bench"       in name: return "mt_bench"
    if "reasoning_recov" in name: return "reasoning_recovery"
    if "tool_calling"   in name: return "tool_calling"
    if "code_consistency" in name: return "code_consistency"
    if "code_thinking"  in name: return "code_thinking"
    if "code_writing"   in name: return "code_writing"
    return "unknown"


def parse_gsm8k(text):
    models, rows = [], []
    for m in re.finditer(r'MODEL:\s+(\S+)', text):
        models.append(m.group(1))
    acc   = re.findall(r'Accuracy\s*:\s*(\d+)/(\d+)\s*\(([0-9.]+)%\)', text)
    lat   = re.findall(r'Avg latency\s*:\s*([0-9.]+)\s*ms', text)
    tok   = re.findall(r'Avg total tokens\s*:\s*([0-9.]+)', text)
    for i, name in enumerate(models):
        rows.append({
            "Model":       name,
            "Accuracy %":  float(acc[i][2]) if i < len(acc) else 0,
            "Correct":     int(acc[i][0])   if i < len(acc) else 0,
            "Total":       int(acc[i][1])   if i < len(acc) else 0,
            "Avg Latency": float(lat[i])    if i < len(lat) else 0,
            "Avg Tokens":  float(tok[i])    if i < len(tok) else 0,
        })
    return pd.DataFrame(rows), "Accuracy %", "GSM8K Accuracy (%)"


def parse_mt_bench(text):
    rows = []
    for m in re.finditer(
        r'MODEL:\s+(\S+).*?Overall score\s*:\s*([0-9.]+)', text, re.DOTALL
    ):
        name, score = m.group(1), float(m.group(2))
        t1 = re.search(r'Turn 1 avg\s*:\s*([0-9.]+)', text[m.start():m.start()+500])
        t2 = re.search(r'Turn 2 avg\s*:\s*([0-9.]+)', text[m.start():m.start()+500])
        rows.append({
            "Model":       name,
            "Overall /10": score,
            "Turn 1 /10":  float(t1.group(1)) if t1 else 0,
            "Turn 2 /10":  float(t2.group(1)) if t2 else 0,
        })

    # Category scores
    cats = ["Writing","Roleplay","Reasoning","Math","Coding","Extraction","STEM","Humanities"]
    cat_data = {}
    for cat in cats:
        scores = re.findall(rf'{cat}\s+.*?([0-9.]+)/10', text)
        if scores:
            cat_data[cat] = [float(s) for s in scores]

    df = pd.DataFrame(rows)
    return df, "Overall /10", "MT-Bench Overall Score (/10)"


def parse_reasoning_recovery(text):
    rows = []
    for m in re.finditer(
        r'MODEL:\s+(\S+).*?Overall accuracy\s*:\s*(\d+)/(\d+)\s*\(([0-9.]+)%\)',
        text, re.DOTALL
    ):
        name = m.group(1)
        block_start = m.start()
        block = text[block_start:block_start+600]
        clean  = re.search(r'Solved cleanly\s*:\s*(\d+)', block)
        recov  = re.search(r'Self-corrected.*?:\s*(\d+)', block)
        false_ = re.search(r'False recovery\s*:\s*(\d+)', block)
        missed = re.search(r'Wrong, never noticed\s*:\s*(\d+)', block)
        rows.append({
            "Model":       name,
            "Accuracy %":  float(m.group(4)),
            "Correct":     int(m.group(2)),
            "Clean":       int(clean.group(1))  if clean  else 0,
            "Recovered":   int(recov.group(1))  if recov  else 0,
            "False Recov": int(false_.group(1)) if false_ else 0,
            "Missed":      int(missed.group(1)) if missed else 0,
        })
    return pd.DataFrame(rows), "Accuracy %", "Reasoning Recovery Accuracy (%)"


def parse_tool_calling(text):
    rows = []
    for m in re.finditer(
        r'MODEL:\s+(\S+).*?Overall score\s*:\s*(\d+)/(\d+)\s*\(([0-9.]+)%\)',
        text, re.DOTALL
    ):
        name  = m.group(1)
        block = text[m.start():m.start()+800]
        corr  = re.search(r'Correct tool sel\.\s*:\s*(\d+)', block)
        part  = re.search(r'Partial tool sel\.\s*:\s*(\d+)', block)
        wrong = re.search(r'Wrong tool sel\.\s*:\s*(\d+)', block)
        param = re.search(r'Avg param accuracy\s*:\s*([0-9.]+)%', block)
        rows.append({
            "Model":          name,
            "Score %":        float(m.group(4)),
            "Total Score":    f"{m.group(2)}/{m.group(3)}",
            "Correct Tools":  int(corr.group(1))  if corr  else 0,
            "Partial":        int(part.group(1))  if part  else 0,
            "Wrong":          int(wrong.group(1)) if wrong else 0,
            "Param Acc %":    float(param.group(1)) if param else 0,
        })
    return pd.DataFrame(rows), "Score %", "Tool Calling Score (%)"


def parse_code_writing(text):
    rows = []
    for m in re.finditer(
        r'MODEL:\s+(\S+).*?Avg score\s*:\s*([0-9.]+)/10',
        text, re.DOTALL
    ):
        name  = m.group(1)
        score = float(m.group(2))
        block = text[m.start():m.start()+600]
        lat   = re.search(r'Avg latency\s*:\s*([0-9.]+)', block)
        tok   = re.search(r'Avg tokens\s*:\s*([0-9.]+)', block)
        rows.append({
            "Model":       name,
            "Avg Score /10": score,
            "Avg Latency": float(lat.group(1)) if lat else 0,
            "Avg Tokens":  float(tok.group(1)) if tok else 0,
        })
    return pd.DataFrame(rows), "Avg Score /10", "Code Writing Score (/10)"


def parse_code_consistency(text):
    rows = []
    for m in re.finditer(
        r'MODEL:\s+(\S+).*?Avg mean score\s*:\s*([0-9.]+)/10.*?'
        r'Avg std deviation\s*:\s*([0-9.]+).*?Avg consistency\s*:\s*([0-9.]+)%',
        text, re.DOTALL
    ):
        rows.append({
            "Model":         m.group(1),
            "Mean Score /10": float(m.group(2)),
            "Std Dev":       float(m.group(3)),
            "Consistency %": float(m.group(4)),
        })
    return pd.DataFrame(rows), "Consistency %", "Consistency Score (%)"


def parse_report(filepath):
    text = Path(filepath).read_text(encoding="utf-8")
    btype = detect_benchmark_type(Path(filepath).name)
    parsers = {
        "gsm8k":              parse_gsm8k,
        "mt_bench":           parse_mt_bench,
        "reasoning_recovery": parse_reasoning_recovery,
        "tool_calling":       parse_tool_calling,
        "code_writing":       parse_code_writing,
        "code_thinking":      parse_code_writing,
        "code_consistency":   parse_code_consistency,
    }
    if btype in parsers:
        return parsers[btype](text), btype, text
    return None, btype, text

# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def bar_chart(df, score_col, title):
    if df is None or df.empty or "Model" not in df.columns:
        return None
    colors = [model_color(m) for m in df["Model"]]
    fig = go.Figure(go.Bar(
        x=df["Model"],
        y=df[score_col],
        marker_color=colors,
        text=[f"{v:.1f}" for v in df[score_col]],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        yaxis_title=score_col,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        height=350,
        margin=dict(t=50, b=20),
    )
    return fig


def radar_chart(df, score_cols, title):
    if df is None or df.empty:
        return None
    fig = go.Figure()
    for _, row in df.iterrows():
        values = [row[c] for c in score_cols if c in df.columns]
        labels = [c for c in score_cols if c in df.columns]
        values += [values[0]]
        labels += [labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=labels,
            fill="toself",
            name=row["Model"],
            line_color=model_color(row["Model"]),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title=title,
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        height=400,
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def build_command(benchmark_name, consistency_n=3):
    cfg = BENCHMARKS[benchmark_name]
    script = str(BENCH_DIR / cfg["script"])
    cmd = [sys.executable, "-u", script]  # -u = unbuffered stdout, lines appear immediately
    modes = cfg["modes"]
    if "thinking"     in modes: cmd.append("--thinking")
    if "consistency"  in modes: cmd += ["--consistency", str(consistency_n)]
    return cmd


def stream_to_queue(cmd, q):
    """Run process in background thread, putting lines into a queue."""
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(BENCH_DIR), env=env,
        )
        for line in proc.stdout:
            q.put(line)
        proc.wait()
    except Exception as e:
        q.put(f"\nERROR: {e}\n")
    finally:
        q.put(None)  # sentinel — signals completion

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RUN BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════

def page_run():
    st.header("Run Benchmarks")

    # ── Persistent state init ─────────────────────────────────────────────────
    for key, default in [
        ("bench_running", False),
        ("bench_output",  []),
        ("bench_name",    ""),
        ("bench_queue",   None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Drain queue → output list (safe: only main thread touches session_state)
    q = st.session_state.bench_queue
    if q is not None:
        while True:
            try:
                line = q.get_nowait()
            except queue_module.Empty:
                break
            if line is None:                         # sentinel: process finished
                st.session_state.bench_running = False
                st.session_state.bench_queue   = None
                break
            st.session_state.bench_output.append(line)

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        benchmark = st.selectbox(
            "Select benchmark", list(BENCHMARKS.keys()),
            disabled=st.session_state.bench_running,
        )
    with col2:
        consistency_n = 1
        if "Consistency" in benchmark:
            consistency_n = st.number_input(
                "Runs per problem", min_value=2, max_value=10, value=3,
                disabled=st.session_state.bench_running,
            )

    cfg         = BENCHMARKS[benchmark]
    script_path = BENCH_DIR / cfg["script"]
    if not script_path.exists():
        st.error(f"Script not found: {cfg['script']}")
        return

    st.caption(f"Script: `{cfg['script']}`  •  Mode: `{', '.join(cfg['modes'])}`")

    # ── Start / running button ────────────────────────────────────────────────
    if st.session_state.bench_running:
        st.button("⏳  Running…", disabled=True, use_container_width=True)
    else:
        if st.button("▶  Run Benchmark", type="primary", use_container_width=True):
            cmd = build_command(benchmark, consistency_n)
            q   = queue_module.Queue()
            st.session_state.bench_queue   = q
            st.session_state.bench_output  = [f"$ {' '.join(cmd)}\n\n"]
            st.session_state.bench_running = True
            st.session_state.bench_name    = benchmark
            threading.Thread(target=stream_to_queue, args=(cmd, q), daemon=True).start()
            # Do NOT call st.rerun() here — let Streamlit's natural rerun continue so
            # st.code() below renders immediately with the cleared output in this pass.

    # ── Output display — always rendered at fixed position so reruns update in-place
    if not st.session_state.bench_running and st.session_state.bench_output:
        st.success(f"✅  Finished: **{st.session_state.bench_name}**")

    output_text = "".join(st.session_state.bench_output[-80:]) if st.session_state.bench_output else ""
    st.code(output_text, language="text")

    if not st.session_state.bench_running and st.session_state.bench_output:
        if st.button("🗑  Clear output"):
            st.session_state.bench_output = []

    # ── Auto-refresh every second while running ───────────────────────────────
    if st.session_state.bench_running:
        time.sleep(1.0)
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: VIEW RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def page_results():
    st.header("View Results")

    files = sorted(RESULTS_DIR.glob("*.txt"), reverse=True)
    if not files:
        st.info("No results yet. Run a benchmark first.")
        return

    # File picker
    file_labels = []
    for f in files:
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        btype = detect_benchmark_type(f.name).replace("_", " ").title()
        file_labels.append(f"{mtime}  ·  {btype}  ·  {f.name}")

    chosen_idx = st.selectbox("Select report", range(len(files)), format_func=lambda i: file_labels[i])
    chosen_file = files[chosen_idx]

    result, btype, raw_text = parse_report(chosen_file)

    if result is None:
        st.warning("Could not parse this report format.")
        with st.expander("Raw report"):
            st.text(raw_text)
        return

    df, score_col, chart_title = result

    # ── Metrics row ────────────────────────────────────────────────────────────
    if df is not None and not df.empty and "Model" in df.columns and score_col in df.columns:
        st.subheader("Summary")
        cols = st.columns(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            with cols[i]:
                st.metric(row["Model"], f"{row[score_col]:.1f}")

        # ── Bar chart ──────────────────────────────────────────────────────────
        fig = bar_chart(df, score_col, chart_title)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # ── Data table ────────────────────────────────────────────────────────
        st.subheader("Full Table")
        st.dataframe(df.set_index("Model"), use_container_width=True)

        # ── Extra charts by type ──────────────────────────────────────────────
        if btype == "mt_bench" and {"Turn 1 /10", "Turn 2 /10"}.issubset(df.columns):
            fig2 = radar_chart(df, ["Turn 1 /10", "Turn 2 /10"], "Turn 1 vs Turn 2")
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)

        if btype == "reasoning_recovery":
            stack_cols = ["Clean", "Recovered", "False Recov", "Missed"]
            if all(c in df.columns for c in stack_cols):
                st.subheader("Recovery Breakdown")
                fig3 = px.bar(
                    df.melt(id_vars="Model", value_vars=stack_cols,
                            var_name="Type", value_name="Count"),
                    x="Model", y="Count", color="Type", barmode="stack",
                    title="Recovery Types per Model",
                    template="plotly_dark",
                )
                st.plotly_chart(fig3, use_container_width=True)

        if btype == "code_consistency" and "Std Dev" in df.columns:
            st.subheader("Consistency vs Score")
            fig4 = px.scatter(
                df, x="Std Dev", y="Mean Score /10", text="Model",
                title="Mean Score vs Std Dev (lower StdDev = more consistent)",
                template="plotly_dark",
                color="Model",
                color_discrete_map={m: model_color(m) for m in df["Model"]},
            )
            fig4.update_traces(textposition="top center", marker_size=12)
            st.plotly_chart(fig4, use_container_width=True)

        if btype == "tool_calling" and "Correct Tools" in df.columns:
            st.subheader("Tool Selection Breakdown")
            fig5 = px.bar(
                df.melt(id_vars="Model", value_vars=["Correct Tools", "Partial", "Wrong"],
                        var_name="Type", value_name="Count"),
                x="Model", y="Count", color="Type", barmode="stack",
                title="Tool Selection Accuracy",
                template="plotly_dark",
                color_discrete_map={"Correct Tools":"#76B900","Partial":"#FF6B35","Wrong":"#E84040"},
            )
            st.plotly_chart(fig5, use_container_width=True)

    # ── Raw report ─────────────────────────────────────────────────────────────
    with st.expander("Raw report text"):
        st.text(raw_text)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: COMPARE
# ══════════════════════════════════════════════════════════════════════════════

def page_compare():
    st.header("Compare Reports")

    files = sorted(RESULTS_DIR.glob("*.txt"), reverse=True)
    if len(files) < 2:
        st.info("Need at least 2 result files to compare.")
        return

    def label(f):
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        btype = detect_benchmark_type(f.name).replace("_", " ").title()
        return f"{mtime}  ·  {btype}  ·  {f.name}"

    col1, col2 = st.columns(2)
    with col1:
        idx_a = st.selectbox("Report A", range(len(files)), format_func=lambda i: label(files[i]), key="cmp_a")
    with col2:
        idx_b = st.selectbox("Report B", range(len(files)), format_func=lambda i: label(files[i]),
                             key="cmp_b", index=min(1, len(files)-1))

    res_a, btype_a, _ = parse_report(files[idx_a])
    res_b, btype_b, _ = parse_report(files[idx_b])

    if res_a is None or res_b is None:
        st.warning("Could not parse one or both reports.")
        return

    df_a, score_col_a, title_a = res_a
    df_b, score_col_b, title_b = res_b

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"A: {files[idx_a].name}")
        if df_a is not None and not df_a.empty:
            st.dataframe(df_a.set_index("Model"), use_container_width=True)
            fig = bar_chart(df_a, score_col_a, title_a)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"B: {files[idx_b].name}")
        if df_b is not None and not df_b.empty:
            st.dataframe(df_b.set_index("Model"), use_container_width=True)
            fig = bar_chart(df_b, score_col_b, title_b)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # Combined comparison if same score column
    if (score_col_a == score_col_b and df_a is not None and df_b is not None
            and not df_a.empty and not df_b.empty):
        st.subheader("Side-by-side Model Scores")
        merged = pd.merge(
            df_a[["Model", score_col_a]].rename(columns={score_col_a: "Report A"}),
            df_b[["Model", score_col_b]].rename(columns={score_col_b: "Report B"}),
            on="Model", how="outer",
        ).fillna(0)

        fig = go.Figure()
        fig.add_bar(name="Report A", x=merged["Model"], y=merged["Report A"],
                    marker_color="#4C9BE8", text=[f"{v:.1f}" for v in merged["Report A"]],
                    textposition="outside")
        fig.add_bar(name="Report B", x=merged["Model"], y=merged["Report B"],
                    marker_color="#FF6B35", text=[f"{v:.1f}" for v in merged["Report B"]],
                    textposition="outside")
        fig.update_layout(
            barmode="group", title="Report A vs Report B",
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#fafafa", height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: AGGREGATED RUNS
# ══════════════════════════════════════════════════════════════════════════════

BTYPE_LABELS = {
    "gsm8k":              "GSM8K Math",
    "mt_bench":           "MT-Bench Chat",
    "reasoning_recovery": "Reasoning Recovery",
    "tool_calling":       "Tool Calling",
    "code_writing":       "Code Writing — Standard",
    "code_thinking":      "Code Writing — Thinking",
    "code_consistency":   "Code Writing — Consistency",
}

def collect_runs(btype):
    """Return list of (datetime, df, score_col) sorted oldest→newest for a benchmark type."""
    files = sorted(RESULTS_DIR.glob("*.txt"))
    runs  = []
    for f in files:
        if detect_benchmark_type(f.name) != btype:
            continue
        result, _, _ = parse_report(f)
        if result is None:
            continue
        df, score_col, _ = result
        if df is None or df.empty or "Model" not in df.columns:
            continue
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        runs.append({"file": f, "time": mtime, "df": df, "score_col": score_col})
    return runs


def build_aggregated_df(runs, score_col):
    """Combine all runs into a long-form DataFrame with Run #, Model, Score, Date."""
    rows = []
    for i, run in enumerate(runs, 1):
        for _, row in run["df"].iterrows():
            if score_col in run["df"].columns:
                rows.append({
                    "Run":   i,
                    "Date":  run["time"].strftime("%m/%d %H:%M"),
                    "Model": row["Model"],
                    "Score": row[score_col],
                    "File":  run["file"].name,
                })
    return pd.DataFrame(rows)


def page_aggregated():
    st.header("Aggregated Runs")
    st.caption("Select a benchmark type to view trends and statistics across all saved runs.")

    files = list(RESULTS_DIR.glob("*.txt"))
    if not files:
        st.info("No results yet. Run a benchmark first.")
        return

    # Find which benchmark types have ≥1 file
    available = {}
    for f in files:
        bt = detect_benchmark_type(f.name)
        available[bt] = available.get(bt, 0) + 1

    type_options = {
        BTYPE_LABELS.get(bt, bt): bt
        for bt, count in sorted(available.items(), key=lambda x: -x[1])
    }

    col1, col2 = st.columns([3, 1])
    with col1:
        chosen_label = st.selectbox("Benchmark type", list(type_options.keys()))
    chosen_btype = type_options[chosen_label]

    runs = collect_runs(chosen_btype)
    with col2:
        st.metric("Total runs found", len(runs))

    if not runs:
        st.warning("No parseable result files found for this benchmark type.")
        return

    score_col = runs[0]["score_col"]
    agg_df    = build_aggregated_df(runs, score_col)
    models    = agg_df["Model"].unique().tolist()

    if agg_df.empty:
        st.warning("Could not extract scores from these files.")
        return

    # ── Summary metrics ────────────────────────────────────────────────────────
    st.subheader("Overall Statistics Across All Runs")
    cols = st.columns(len(models))
    for i, model in enumerate(models):
        model_df = agg_df[agg_df["Model"] == model]["Score"]
        mean = model_df.mean()
        std  = model_df.std() if len(model_df) > 1 else 0
        best = model_df.max()
        with cols[i]:
            st.metric(model, f"{mean:.2f}", delta=f"best {best:.1f}  σ {std:.2f}")

    st.divider()

    # ── Score trend over runs ─────────────────────────────────────────────────
    st.subheader(f"Score Trend — {score_col} per Run")
    fig_line = go.Figure()
    for model in models:
        mdf = agg_df[agg_df["Model"] == model].sort_values("Run")
        fig_line.add_trace(go.Scatter(
            x=mdf["Run"],
            y=mdf["Score"],
            mode="lines+markers",
            name=model,
            line_color=model_color(model),
            text=mdf["Date"],
            hovertemplate="Run %{x}<br>%{y:.2f}<br>%{text}<extra></extra>",
            marker=dict(size=8),
        ))
    fig_line.update_layout(
        xaxis_title="Run #",
        yaxis_title=score_col,
        xaxis=dict(tickmode="linear", dtick=1),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ── Distribution box plot ─────────────────────────────────────────────────
    col_box, col_bar = st.columns(2)

    with col_box:
        st.subheader("Score Distribution")
        fig_box = go.Figure()
        for model in models:
            mdf = agg_df[agg_df["Model"] == model]
            fig_box.add_trace(go.Box(
                y=mdf["Score"],
                name=model,
                marker_color=model_color(model),
                boxmean="sd",
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
            ))
        fig_box.update_layout(
            yaxis_title=score_col,
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#fafafa",
            height=380,
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with col_bar:
        st.subheader("Mean ± Std Dev")
        means, stds, names, colors = [], [], [], []
        for model in models:
            mdf = agg_df[agg_df["Model"] == model]["Score"]
            means.append(mdf.mean())
            stds.append(mdf.std() if len(mdf) > 1 else 0)
            names.append(model)
            colors.append(model_color(model))

        fig_err = go.Figure(go.Bar(
            x=names,
            y=means,
            error_y=dict(type="data", array=stds, visible=True),
            marker_color=colors,
            text=[f"{m:.2f}" for m in means],
            textposition="outside",
        ))
        fig_err.update_layout(
            yaxis_title=f"Mean {score_col}",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#fafafa",
            height=380,
        )
        st.plotly_chart(fig_err, use_container_width=True)

    # ── Best vs worst run ─────────────────────────────────────────────────────
    st.subheader("Best vs Worst Run (by average score across models)")
    run_avgs = agg_df.groupby("Run")["Score"].mean().reset_index()
    run_avgs.columns = ["Run", "Avg Score"]

    if len(run_avgs) >= 2:
        best_run  = run_avgs.loc[run_avgs["Avg Score"].idxmax(), "Run"]
        worst_run = run_avgs.loc[run_avgs["Avg Score"].idxmin(), "Run"]
        best_file  = runs[int(best_run)  - 1]["file"].name
        worst_file = runs[int(worst_run) - 1]["file"].name

        c1, c2 = st.columns(2)
        with c1:
            st.success(f"**Best run: #{int(best_run)}**  avg={run_avgs.loc[run_avgs['Run']==best_run,'Avg Score'].values[0]:.2f}\n\n`{best_file}`")
        with c2:
            st.error(f"**Worst run: #{int(worst_run)}**  avg={run_avgs.loc[run_avgs['Run']==worst_run,'Avg Score'].values[0]:.2f}\n\n`{worst_file}`")

    # ── Per-run scores table ───────────────────────────────────────────────────
    st.subheader("All Runs — Score Table")
    pivot = agg_df.pivot_table(index=["Run","Date"], columns="Model", values="Score").reset_index()
    pivot["Avg"] = pivot[models].mean(axis=1).round(2)
    pivot = pivot.sort_values("Run")
    st.dataframe(pivot.set_index("Run"), use_container_width=True)

    # ── Run-over-run delta ─────────────────────────────────────────────────────
    if len(runs) >= 2:
        st.subheader("Run-over-Run Change")
        fig_delta = go.Figure()
        for model in models:
            mdf = agg_df[agg_df["Model"] == model].sort_values("Run")
            deltas = mdf["Score"].diff().fillna(0).tolist()
            runs_x = mdf["Run"].tolist()
            colors_delta = ["#76B900" if d >= 0 else "#E84040" for d in deltas[1:]]
            fig_delta.add_trace(go.Bar(
                x=runs_x[1:],
                y=deltas[1:],
                name=model,
                marker_color=colors_delta,
                showlegend=True,
            ))
        fig_delta.add_hline(y=0, line_dash="dash", line_color="#888")
        fig_delta.update_layout(
            xaxis_title="Run #",
            yaxis_title=f"Δ {score_col}",
            xaxis=dict(tickmode="linear", dtick=1),
            barmode="group",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#fafafa",
            height=320,
        )
        st.plotly_chart(fig_delta, use_container_width=True)

    # ── File list ─────────────────────────────────────────────────────────────
    with st.expander(f"Show all {len(runs)} files included"):
        for i, run in enumerate(runs, 1):
            st.caption(f"Run {i} — {run['time'].strftime('%Y-%m-%d %H:%M')} — `{run['file'].name}`")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def page_history():
    st.header("Results History")

    files = sorted(RESULTS_DIR.glob("*.txt"), reverse=True)
    if not files:
        st.info("No results yet.")
        return

    rows = []
    for f in files:
        btype = detect_benchmark_type(f.name).replace("_", " ").title()
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        size  = f"{f.stat().st_size // 1024} KB"
        rows.append({"File": f.name, "Type": btype, "Date": mtime, "Size": size})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"{len(files)} result files in `{RESULTS_DIR}`")

# ══════════════════════════════════════════════════════════════════════════════
#  APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="LLM Benchmark Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stMetricValue"]  { font-size: 1.8rem; font-weight: 700; }
    [data-testid="stMetricLabel"]  { font-size: 0.9rem; color: #aaa; }
    .block-container               { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📊 LLM Benchmarks")
    st.caption("AI-End-to-End Project")
    st.divider()
    page = st.radio(
        "Navigation",
        ["▶  Run Benchmarks", "📈  View Results", "🔁  Aggregated Runs", "🗂  History"],
        label_visibility="collapsed",
    )
    st.divider()
    files = sorted(RESULTS_DIR.glob("*.txt"), reverse=True)
    st.caption(f"**{len(files)}** reports saved")
    if files:
        latest = files[0]
        mtime  = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%b %d, %H:%M")
        st.caption(f"Latest: {latest.name[:30]}…\n{mtime}")

if   page == "▶  Run Benchmarks":  page_run()
elif page == "📈  View Results":    page_results()
elif page == "🔁  Aggregated Runs": page_aggregated()
elif page == "🗂  History":         page_history()
