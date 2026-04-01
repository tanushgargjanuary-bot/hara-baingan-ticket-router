# 🎫 Customer Support Ticket Router — OpenEnv Environment

An **OpenEnv** environment that simulates a customer support inbox for the
**Meta PyTorch Hackathon**. An AI agent reads realistic support tickets and
must triage them — classifying priority, routing to the right department,
drafting a helpful first response, and extracting action items. A deterministic
grader scores every action from **0.0 to 1.0** with meaningful partial credit.

---

## Motivation

Customer support triage is one of the highest-value real-world applications of
LLMs: every day, millions of tickets need to be prioritised, routed, and
answered quickly. This environment provides a **reproducible, quantitative
benchmark** for how well an AI agent handles that workflow across three
difficulty tiers.

---

## Tasks

| # | Name | Difficulty | Graded Fields | Scoring |
|---|------|-----------|---------------|---------|
| 1 | **Priority Classification** | Easy | `priority` | exact = 1.0 · adjacent = 0.5 · wrong = 0.0 |
| 2 | **Department Routing** | Medium | `priority`, `department` | 50 % priority + 50 % department (exact match) |
| 3 | **Full Response** | Hard | `priority`, `department`, `response`, `action_items` | 25 % pri + 25 % dept + 30 % keyword-overlap + 20 % action-item match |

Additional **penalties** (−0.10 for High↔Low confusion, −0.05 for
cross-category routing errors) prevent inflated scores on random guessing.

---

## Observation Space (what the agent sees)

| Field | Type | Description |
|-------|------|-------------|
| `tickets` | `List[TicketData]` | 1-3 tickets with `ticket_id`, `customer_name`, `subject`, `body`, `timestamp` |
| `task_id` | `int` | Active task (1 / 2 / 3) |
| `task_description` | `str` | Human-readable task instructions |
| `tickets_remaining` | `int` | Tickets still to be processed |
| `last_score` | `TicketScore?` | Detailed grading breakdown of previous step |
| `done` | `bool` | `True` once all tickets in the episode are handled |
| `reward` | `float` | Running mean task-weighted score (0.0 – 1.0) |

## Action Space (what the agent submits)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ticket_id` | `str` | ✅ | Which ticket this action addresses |
| `priority` | `str` | ✅ | `Low` / `Medium` / `High` |
| `department` | `str` | Task ≥ 2 | `Tech` / `Billing` / `General` / `Escalation` |
| `response` | `str` | Task 3 | Helpful first response to the customer |
| `action_items` | `List[str]` | Task 3 | Action items for the support team |

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r server/requirements.txt
```

### 2 — Start the environment server

```bash
python -m server.app --port 8000
```

### 3 — Run the baseline agent

```bash
export OPENAI_API_KEY="sk-..."
python client.py --base-url http://localhost:8000 --model gpt-4o-mini --episodes 5 --tasks 1 2 3
```

### Docker

```bash
docker build -t support-ticket-router .
docker run -p 8000:8000 support-ticket-router
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start a new episode (returns 1-3 tickets) |
| `POST` | `/step` | Submit action for one ticket |
| `GET`  | `/state` | Current environment state |
| `GET`  | `/tasks` | Metadata for all 3 tasks |
| `POST` | `/grader` | Grader scores for the current episode |
| `POST` | `/baseline` | Run OpenAI baseline inference server-side |

---

## Baseline Scores (gpt-4o-mini, 10 episodes)

| Task | Mean Score | Notes |
|------|-----------|-------|
| 1 — Priority | ~0.85 | Occasionally confuses Medium/High |
| 2 — Routing | ~0.78 | General vs Escalation is tricky |
| 3 — Full Response | ~0.72 | Response quality varies; action items need specificity |

*Scores are approximate and vary with sampling randomness.*

---

## Project Structure

```
.
├── models.py                                # Pydantic Action / Observation models
├── client.py                                # Baseline inference script (OpenAI)
├── openenv.yaml                             # OpenEnv metadata & task definitions
├── .env.example                             # Template for environment variables
├── README.md
└── server/
    ├── __init__.py                          # Makes server a Python package
    ├── app.py                               # FastAPI app + custom endpoints
    ├── support_ticket_router_environment.py # Core environment & grader logic
    └── requirements.txt
```

---

## Design Decisions

1. **Partial credit everywhere** — Adjacent priority guesses score 0.5;
   keyword overlap gives continuous signal for response quality; action-item
   matching uses word-overlap so paraphrasing is accepted.
2. **Deterministic grading** — No LLM-as-a-judge. All graders are pure
   functions of the action and the ground-truth labels, making results
   perfectly reproducible.
3. **Penalties, not just zeroes** — Egregiously wrong actions (e.g. marking a
   hacked-account ticket as Low priority) receive a small negative adjustment,
   discouraging random guessing.
4. **15 diverse tickets** — Cover Tech, Billing, General, and Escalation
   departments with realistic customer language and varying urgency.

---

## License

MIT
