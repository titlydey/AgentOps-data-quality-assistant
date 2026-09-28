# AgentOps Data Quality Assistant

A small multi-agent workflow (LangGraph) that answers questions by routing to specialist agents:

| Agent | What it does | Tech |
|---|---|---|
| quality | Validates a CSV: missing values, duplicates, invalid dates, anomalies | pandas, scikit-learn IsolationForest |
| api | Fetches live repo stats as an external tool | REST API (GitHub), input validation |
| knowledge | Answers from a small knowledge base (RAG) | ChromaDB, custom offline embeddings |

A guardrail node blocks prompt-injection attempts and long inputs, and redacts emails and phone numbers from answers.
If `ANTHROPIC_API_KEY` is set, the final answer is written by an LLM using only the tool result; otherwise it returns the raw tool result.

## Run
```
pip install -r requirements.txt
python scripts/make_sample.py      # creates data/sample.csv with injected issues
pytest                              # routing eval, tools, RAG, guardrails, API tests
uvicorn app.main:api --reload       # then POST /ask {"question": "Are there anomalies in the dataset?"}
docker build -t agentops-dq . && docker run -p 8000:8000 agentops-dq
```

## Evaluate
`tests/test_agent.py` holds a 10-question routing eval (must stay >= 90%) and checks the quality tool against issues injected with known positions.

## Limitations / next steps
Routing is rule-based; an LLM router, memory and a larger eval set are natural extensions. Embeddings are hashed word vectors, not a neural model.
