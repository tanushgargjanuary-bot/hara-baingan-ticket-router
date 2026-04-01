"""
Pydantic models for the Customer Support Ticket Router environment.

Defines the Action (agent input) and Observation (environment output) schemas
used across all three tasks (priority classification, department routing, full response).
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation


# ---------------------------------------------------------------------------
# Sub-models (shared data structures)
# ---------------------------------------------------------------------------

class TicketData(BaseModel):
    """A single support ticket as presented to the agent."""

    ticket_id: str = Field(..., description="Unique ticket identifier, e.g. TKT-001")
    customer_name: str = Field(..., description="Name of the customer")
    subject: str = Field(..., description="One-line subject of the ticket")
    body: str = Field(..., description="Full body / description of the issue")
    timestamp: str = Field(..., description="ISO-8601 timestamp of ticket creation")


class TicketScore(BaseModel):
    """Per-ticket grading breakdown returned in observation metadata."""

    ticket_id: str = ""
    priority_score: float = 0.0
    department_score: float = 0.0
    response_score: float = 0.0
    action_items_score: float = 0.0
    penalty: float = 0.0
    task_weighted_score: float = 0.0


# ---------------------------------------------------------------------------
# Action – what the agent submits each step
# ---------------------------------------------------------------------------

class SupportTicketRouterAction(Action):
    """
    The agent's response for a single ticket.

    * Task 1 (Easy)   – only `priority` is graded.
    * Task 2 (Medium) – `priority` + `department` are graded.
    * Task 3 (Hard)   – all four fields are graded.
    """

    ticket_id: str = Field(
        ..., description="ID of the ticket this action addresses"
    )
    priority: str = Field(
        "Medium",
        description="Priority classification: Low | Medium | High",
    )
    department: str = Field(
        "General",
        description="Target department: Tech | Billing | General | Escalation",
    )
    response: str = Field(
        "",
        description="Helpful first-response message to the customer",
    )
    action_items: List[str] = Field(
        default_factory=list,
        description="Extracted action items for the support team",
    )


# ---------------------------------------------------------------------------
# Observation – what the environment returns
# ---------------------------------------------------------------------------

class SupportTicketRouterObservation(Observation):
    """
    Returned by reset() and step().

    On reset  → tickets are populated, scores are empty.
    On step   → feedback for the just-processed ticket is in `last_score`;
                cumulative stats live in metadata.
    """

    tickets: List[TicketData] = Field(
        default_factory=list,
        description="Support tickets for this episode (populated on reset)",
    )
    task_id: int = Field(
        default=1,
        description="Active task (1=Easy, 2=Medium, 3=Hard)",
    )
    task_description: str = Field(
        default="",
        description="Human-readable description of the current task",
    )
    tickets_remaining: int = Field(
        default=0,
        description="How many tickets still need to be processed",
    )
    last_score: Optional[TicketScore] = Field(
        default=None,
        description="Grading breakdown for the most recently submitted ticket",
    )
