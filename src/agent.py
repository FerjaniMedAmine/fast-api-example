"""
Two-agent CI log analysis pipeline.

Agent 1 (classify_agent): reads the full raw log, outputs a structured
                           list of anomalies with a category + severity.
Agent 2 (report_agent):   takes that structured list, orders it high -> low
                           severity, and writes the final Markdown report.

Orchestrated as a 2-node LangGraph.

Usage:
    python log_agent.py path/to/raw_log.txt
"""

import sys
from typing import List, Literal

from pydantic import BaseModel, Field
from langchain.agents import create_agent, AgentState
from langgraph.graph import StateGraph, END


# ---------- 1. Schemas ----------

class Anomaly(BaseModel):
    category: Literal["test_failure", "lint_warning", "deprecation_warning", "build_error", "other"]
    severity: Literal["low", "medium", "high"] = Field(
        description="GitHub does not provide this - you must judge it from the log content"
    )
    file: str = Field(description="File path mentioned in the log, or 'unknown' if none")
    message: str = Field(description="Short description of the issue, in your own words")
    raw_excerpt: str = Field(description="The exact original log line(s) that triggered this")


class AnomalyReport(BaseModel):
    anomalies: List[Anomaly]
    summary: str = Field(description="One or two sentence overall summary of the log's health")


class FinalReport(BaseModel):
    markdown: str = Field(description="The complete Markdown report, anomalies ordered high -> low severity")


# ---------- 2. Agent 1: classify ----------

classify_agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    tools=[],  # no actions needed - the full log is already in the prompt
    system_prompt=(
        "You are a CI/CD log analysis assistant. Read the raw GitHub Actions log "
        "and identify every anomaly: test failures, lint warnings, deprecation "
        "warnings, build errors, or anything else noteworthy. Ignore purely "
        "informational lines (installing dependencies, progress bars, timestamps). "
        "GitHub does not label severity for you - assign 'low', 'medium', or 'high' "
        "yourself based on how serious the issue actually is."
    ),
    response_format=AnomalyReport,
)


# ---------- 3. Agent 2: report ----------

report_agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    tools=[],  # just reformatting what agent 1 already found
    system_prompt=(
        "You receive a JSON list of anomalies already classified with a category "
        "and severity. Order them from high to low severity and write a clean, "
        "readable Markdown report grouped by severity level. Do not invent new "
        "anomalies - only reformat and order what you were given."
    ),
    response_format=FinalReport,
)


# ---------- 4. Outer pipeline state ----------
# AgentState is the same base state each create_agent() uses internally
# (holds "messages"). We extend it here just to carry our own fields
# between the two graph nodes.

class PipelineState(AgentState):
    raw_log: str
    anomaly_report: AnomalyReport
    markdown_report: str


def classify_node(state: PipelineState) -> dict:
    result = classify_agent.invoke({"messages": [{"role": "user", "content": state["raw_log"]}]})
    return {"anomaly_report": result["structured_response"]}


def report_node(state: PipelineState) -> dict:
    payload = state["anomaly_report"].model_dump_json()
    result = report_agent.invoke({"messages": [{"role": "user", "content": payload}]})
    return {"markdown_report": result["structured_response"].markdown}


graph = StateGraph(PipelineState)
graph.add_node("classify", classify_node)
graph.add_node("report", report_node)
graph.set_entry_point("classify")
graph.add_edge("classify", "report")
graph.add_edge("report", END)
app = graph.compile()


# ---------- 5. Run it ----------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python log_agent.py path/to/raw_log.txt")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as f:
        raw_log = f.read()

    final_state = app.invoke({"raw_log": raw_log, "messages": []})

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(final_state["markdown_report"])

    print("Report written to report.md")