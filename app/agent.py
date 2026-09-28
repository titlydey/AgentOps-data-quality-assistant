import os, re
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from app import tools, rag, guardrails

class S(TypedDict, total=False):
    question: str
    blocked: bool
    route: str
    result: object
    answer: str

REPO = re.compile(r"\b[\w.-]+/[\w.-]+\b")

def route(q):
    """Supervisor: pick a specialist agent."""
    ql = q.lower()
    if ("github" in ql or "repo" in ql) and REPO.search(q):
        return "api"
    if any(w in ql for w in ("anomal", "duplicate", "missing", "invalid", "dataset", "csv", "check the data")):
        return "quality"
    return "knowledge"

def guard(s):
    return {"blocked": not guardrails.input_ok(s["question"])}

def quality(s):
    return {"route": "quality", "result": tools.data_quality()}

def api(s):
    try:
        return {"route": "api", "result": tools.github_repo(REPO.search(s["question"]).group())}
    except Exception as e:
        return {"route": "api", "result": {"error": str(e)}}

def knowledge(s):
    return {"route": "knowledge", "result": rag.search(s["question"])}

def llm_answer(q, result):
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"), max_tokens=400,
            messages=[{"role": "user", "content": f"Answer using ONLY this tool result.\nQuestion: {q}\nResult: {result}"}])
        return r.content[0].text
    except Exception:
        return None

def respond(s):
    if s.get("blocked"):
        return {"route": "blocked", "answer": "Request blocked by guardrails."}
    text = llm_answer(s["question"], s["result"]) or f"[{s['route']}] {s['result']}"
    return {"answer": guardrails.redact(text)}

g = StateGraph(S)
for name, fn in (("guard", guard), ("quality", quality), ("api", api), ("knowledge", knowledge), ("respond", respond)):
    g.add_node(name, fn)
g.add_edge(START, "guard")
g.add_conditional_edges("guard", lambda s: "respond" if s["blocked"] else route(s["question"]),
                        {"respond": "respond", "quality": "quality", "api": "api", "knowledge": "knowledge"})
for n in ("quality", "api", "knowledge"):
    g.add_edge(n, "respond")
g.add_edge("respond", END)
graph = g.compile()

def ask(question):
    return graph.invoke({"question": question})
