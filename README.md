# Customer Support Ticket Router

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-0.2.0-7B68EE)](https://github.com/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An [OpenEnv](https://github.com/openenv) environment that simulates a customer support inbox. An AI agent reads realistic support tickets and must triage them — classifying priority, routing to the correct department, drafting a helpful response, and extracting action items. A deterministic grader scores every action from **0.0 to 1.0**.

Built for the **Meta PyTorch Hackathon**.

---

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────┐
│  /reset     │────▶│ 1-3 tickets  │────▶│ Agent processes  │────▶│ /step ×N │
│  (new ep)   │     │ returned     │     │ each ticket      │     │ (grade)  │
└─────────────┘     └──────────────┘     └──────────────────┘     └──────────┘
                                                                        │
                                                                        ▼
                                                               ┌─────────────────┐
                                                               │  Score: 0.0-1.0 │
                                                               │  per ticket     │
                                                               └─────────────────┘
```

1. **Reset** — the environment returns 1-3 random tickets from a pool of 15
2. **Triage** — the agent reads each ticket and submits a structured action
3. **Grade** — each action is scored deterministically against ground-truth labels
4. **Done** — when all tickets are processed, the episode reward is the mean score

## Tasks

Three difficulty tiers, each building on the last:

| Task | Difficulty | What You Do | Graded Fields |
|------|-----------|-------------|---------------|
| **1** | Easy | Classify priority | `priority` |
| **2** | Medium | Classify priority + route to department | `priority`, `department` |
| **3** | Hard | Priority + department + response + action items | All four fields |

### Scoring

| Field | Exact Match | Partial Credit |
|-------|------------|----------------|
| **Priority** | 1.0 | 0.5 for adjacent (Low↔Medium, Medium↔High) |
| **Department** | 1.0 | None — exact or zero |
| **Response** | — | Keyword overlap: fraction of expected keywords found |
| **Action Items** | — | Word overlap: ≥40% shared words counts as a match |

**Penalties** prevent random guessing:
- **−0.10** for High↔Low priority confusion
- **−0.05** for cross-category routing errors (Tech↔Billing, General↔Escalation)

### Example

**Ticket:**
```
ID: TKT-003 | Customer: Carol Davis
Subject: How do I reset my password?
Body: I forgot my password and can't log in. The 'Forgot Password' link
      doesn't seem to be sending me an email.
```

**Expected action (Task 3):**
```json
{
  "ticket_id": "TKT-003",
  "priority": "Low",
  "department": "General",
  "response": "Hi Carol, sorry you're having trouble logging in. I've sent a fresh password reset link to your email. Please check your spam folder if it doesn't arrive within a few minutes.",
  "action_items": [
    "Verify customer email on file",
    "Trigger manual password reset link",
    "Check spam/junk folder"
  ]
}
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r server/requirements.txt
```

### 2. Set up environment variables

Copy the example file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your keys — `GEMINI_API_KEY` for the Gemini baseline client, `OPENAI_API_KEY` for the server-side `/baseline` endpoint.

### 3. Start the server

```bash
python -m server.app --port 8000
```

Visit [http://localhost:8000](http://localhost:8000) to verify it's running. You'll see the API info and available routes.

---

## Running the Baseline Agent

The included `client.py` runs an LLM (OpenAI or Gemini) against the environment and reports scores:

```bash
# With Gemini (default)
python client.py --provider gemini --model gemini-2.0-flash --episodes 5

# With OpenAI
python client.py --provider openai --model gpt-4o-mini --episodes 5
```

Or run the server-side baseline via the API:

```bash
curl -X POST http://localhost:8000/baseline \
  -H "Content-Type: application/json" \
  -d '{"task_id": 3, "model": "gpt-4o-mini", "num_episodes": 3}'
```

### Manual Testing

No API key needed — process tickets yourself in the terminal:

```bash
python test_manual.py
```

---

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | API info and available routes |
| `POST` | `/reset` | Start a new episode — returns 1-3 tickets |
| `POST` | `/step` | Submit an action for one ticket — returns score |
| `GET`  | `/state` | Current environment state |
| `GET`  | `/tasks` | Metadata for all 3 tasks with action schemas |
| `POST` | `/grader` | Get aggregated scores for the current episode |
| `POST` | `/baseline` | Run OpenAI baseline inference server-side |

### Observation (what the agent receives)

| Field | Type | Description |
|-------|------|-------------|
| `tickets` | `List[TicketData]` | 1-3 tickets with `ticket_id`, `customer_name`, `subject`, `body`, `timestamp` |
| `task_id` | `int` | Active task (1 / 2 / 3) |
| `task_description` | `str` | Human-readable task instructions |
| `tickets_remaining` | `int` | Tickets still to be processed |
| `last_score` | `TicketScore?` | Detailed grading breakdown of the previous step |
| `done` | `bool` | `True` once all tickets are handled |
| `reward` | `float` | Running mean task-weighted score (0.0 – 1.0) |

### Action (what the agent submits)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ticket_id` | `str` | Always | Which ticket this action addresses |
| `priority` | `str` | Always | `Low` / `Medium` / `High` |
| `department` | `str` | Task ≥ 2 | `Tech` / `Billing` / `General` / `Escalation` |
| `response` | `str` | Task 3 | Helpful first response to the customer |
| `action_items` | `List[str]` | Task 3 | Action items for the support team |

---

## Project Structure

```
.
├── models.py                                # Pydantic Action / Observation models
├── client.py                                # Baseline inference client (OpenAI + Gemini)
├── test_manual.py                           # Terminal-based manual testing
├── openenv.yaml                             # OpenEnv metadata & task definitions
├── .env.example                             # Environment variable template
├── README.md
└── server/
    ├── __init__.py                          # Makes server a Python package
    ├── app.py                               # FastAPI app + custom endpoints
    ├── support_ticket_router_environment.py # Core environment, 15 tickets, grader logic
    └── requirements.txt
```

---

## Design Decisions

- **Partial credit everywhere** — adjacent priority guesses score 0.5, keyword overlap gives continuous signal for response quality, action-item matching accepts paraphrasing
- **Deterministic grading** — no LLM-as-a-judge; all graders are pure functions of the action and ground-truth labels, making results perfectly reproducible
- **Penalties, not just zeroes** — egregiously wrong actions (e.g. marking a hacked-account ticket as Low priority) receive a small negative adjustment to discourage random guessing
- **15 diverse tickets** — covering Tech, Billing, General, and Escalation departments with realistic customer language and varying urgency

---

## License

MIT
