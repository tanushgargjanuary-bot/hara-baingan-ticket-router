"""
FastAPI application for the Customer Support Ticket Router.

Standard OpenEnv endpoints are created by `create_app()`.
Additional endpoints:
    GET  /tasks    → list of 3 tasks with action schemas
    POST /grader   → deterministic grader scores for the current episode
    POST /baseline → run baseline inference (OpenAI) and return scores
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required") from e

try:
    from ..models import SupportTicketRouterAction, SupportTicketRouterObservation
    from .support_ticket_router_environment import (
        TASK_DESCRIPTIONS,
        SupportTicketRouterEnvironment,
    )
except ImportError:
    from models import SupportTicketRouterAction, SupportTicketRouterObservation
    from server.support_ticket_router_environment import (
        TASK_DESCRIPTIONS,
        SupportTicketRouterEnvironment,
    )

# ------------------------------------------------------------------
# Create core OpenEnv app
# ------------------------------------------------------------------

app = create_app(
    SupportTicketRouterEnvironment,
    SupportTicketRouterAction,
    SupportTicketRouterObservation,
    env_name="support_ticket_router",
    max_concurrent_envs=5,
)

# ------------------------------------------------------------------
# Singleton env reference used by custom endpoints
# ------------------------------------------------------------------
_env_instance: Optional[SupportTicketRouterEnvironment] = None


def _get_env() -> SupportTicketRouterEnvironment:
    """Return (or create) a shared environment instance for custom endpoints."""
    global _env_instance
    if _env_instance is None:
        _env_instance = SupportTicketRouterEnvironment()
    return _env_instance


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Customer Support Ticket Router API",
        "routes": {
            "GET /tasks": "List available support tickets",
            "POST /grader": "Grade a solution against ground truth",
            "POST /baseline": "Run baseline inference on a ticket",
            "POST /reset": "Start a new episode",
            "POST /step": "Submit action for a ticket",
            "GET /state": "Get current environment state",
        },
    }


# ------------------------------------------------------------------
# GET /tasks
# ------------------------------------------------------------------

class TaskInfo(BaseModel):
    task_id: int
    name: str
    description: str
    graded_fields: List[str]
    action_schema: Dict[str, Any]


@app.get("/tasks", response_model=List[TaskInfo])
async def get_tasks():
    """Return metadata for the three tasks including action schemas."""
    schema = SupportTicketRouterAction.model_json_schema()

    tasks = [
        TaskInfo(
            task_id=1,
            name="Priority Classification",
            description=TASK_DESCRIPTIONS[1],
            graded_fields=["priority"],
            action_schema={
                k: v
                for k, v in schema.get("properties", {}).items()
                if k in ("ticket_id", "priority")
            },
        ),
        TaskInfo(
            task_id=2,
            name="Department Routing",
            description=TASK_DESCRIPTIONS[2],
            graded_fields=["priority", "department"],
            action_schema={
                k: v
                for k, v in schema.get("properties", {}).items()
                if k in ("ticket_id", "priority", "department")
            },
        ),
        TaskInfo(
            task_id=3,
            name="Full Response",
            description=TASK_DESCRIPTIONS[3],
            graded_fields=["priority", "department",
                           "response", "action_items"],
            action_schema=schema.get("properties", {}),
        ),
    ]
    return tasks


# ------------------------------------------------------------------
# POST /grader
# ------------------------------------------------------------------

class GraderRequest(BaseModel):
    task_id: int = 3


class GraderResponse(BaseModel):
    episode_id: str
    task_id: int
    scores: List[Dict[str, Any]]
    mean_score: float


@app.post("/grader", response_model=GraderResponse)
async def grader(req: GraderRequest):
    """
    Return deterministic grader scores for the most recent episode.

    If the task_id differs from what was used during the episode, the
    environment re-grades stored actions accordingly.
    """
    env = _get_env()
    result = env.get_episode_scores()
    if not result.get("scores"):
        raise HTTPException(
            status_code=400,
            detail="No episode data. Call /reset then /step first.",
        )
    return GraderResponse(**result)


# ------------------------------------------------------------------
# POST /baseline
# ------------------------------------------------------------------

class BaselineRequest(BaseModel):
    task_id: int = 3
    model: str = "gpt-4o-mini"
    num_episodes: int = 3


class BaselineResponse(BaseModel):
    model: str
    task_id: int
    episodes: List[Dict[str, Any]]
    overall_mean_score: float


@app.post("/baseline", response_model=BaselineResponse)
async def run_baseline(req: BaselineRequest):
    """
    Run baseline inference using the OpenAI API and return scores.
    Reads OPENAI_API_KEY from environment variable.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY environment variable not set.",
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(
            status_code=500, detail="openai package not installed."
        )

    client = OpenAI(api_key=api_key)
    env = _get_env()
    env.set_task(req.task_id)

    all_episodes: List[Dict[str, Any]] = []

    for ep_idx in range(req.num_episodes):
        obs = env.reset()
        episode_actions = []

        for ticket in obs.tickets:
            prompt = _build_prompt(ticket, req.task_id)

            chat_resp = client.chat.completions.create(
                model=req.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a customer support triage AI. "
                            "Respond ONLY with valid JSON (no markdown)."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw = chat_resp.choices[0].message.content.strip()

            # Parse JSON from LLM response
            try:
                # Strip markdown fences if present
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {}

            action = SupportTicketRouterAction(
                ticket_id=ticket.ticket_id,
                priority=parsed.get("priority", "Medium"),
                department=parsed.get("department", "General"),
                response=parsed.get("response", ""),
                action_items=parsed.get("action_items", []),
            )
            step_obs = env.step(action)
            episode_actions.append(
                {
                    "ticket_id": ticket.ticket_id,
                    "action": action.model_dump(),
                    "score": step_obs.last_score.model_dump()
                    if step_obs.last_score
                    else None,
                }
            )

        ep_result = env.get_episode_scores()
        all_episodes.append(
            {
                "episode": ep_idx + 1,
                "actions": episode_actions,
                "mean_score": ep_result["mean_score"],
            }
        )

    overall = round(
        sum(e["mean_score"] for e in all_episodes) / len(all_episodes), 4
    )

    return BaselineResponse(
        model=req.model,
        task_id=req.task_id,
        episodes=all_episodes,
        overall_mean_score=overall,
    )


def _build_prompt(ticket, task_id: int) -> str:
    """Build an LLM prompt for the given ticket and task."""
    base = (
        f"Support Ticket ID: {ticket.ticket_id}\n"
        f"Customer: {ticket.customer_name}\n"
        f"Subject: {ticket.subject}\n"
        f"Body: {ticket.body}\n"
        f"Timestamp: {ticket.timestamp}\n\n"
    )

    if task_id == 1:
        base += (
            'Classify the priority as Low, Medium, or High.\n'
            'Return JSON: {"priority": "..."}\n'
        )
    elif task_id == 2:
        base += (
            'Classify the priority (Low/Medium/High) and route to a department '
            '(Tech/Billing/General/Escalation).\n'
            'Return JSON: {"priority": "...", "department": "..."}\n'
        )
    else:
        base += (
            "Classify the priority (Low/Medium/High), route to a department "
            "(Tech/Billing/General/Escalation), write a helpful first response "
            "to the customer, and extract action items for the support team.\n"
            "Return JSON:\n"
            "{\n"
            '  "priority": "...",\n'
            '  "department": "...",\n'
            '  "response": "...",\n'
            '  "action_items": ["...", "..."]\n'
            "}\n"
        )
    return base


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
