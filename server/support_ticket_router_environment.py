"""
Core environment logic for the Customer Support Ticket Router.

Lifecycle
---------
1. Caller sets `task_id` (1/2/3) then calls reset().
2. reset() samples 1-3 tickets and returns them as an observation.
3. The agent calls step() once per ticket (order doesn't matter).
4. After all tickets are processed the episode is done.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        SupportTicketRouterAction,
        SupportTicketRouterObservation,
        TicketData,
        TicketScore,
    )
except ImportError:
    from models import (
        SupportTicketRouterAction,
        SupportTicketRouterObservation,
        TicketData,
        TicketScore,
    )


# ===================================================================
# Ground-truth ticket database (15 diverse tickets)
# ===================================================================

TICKET_DATABASE: List[Dict[str, Any]] = [
    # 1 ---------------------------------------------------------------
    {
        "ticket_id": "TKT-001",
        "customer_name": "Alice Johnson",
        "subject": "My laptop won't turn on",
        "body": (
            "I pressed the power button multiple times but nothing happens. "
            "No lights, no fan noise, completely dead. I purchased it two weeks ago."
        ),
        "timestamp": "2025-06-01T09:15:00Z",
        "priority": "High",
        "department": "Tech",
        "response_keywords": [
            "power", "troubleshoot", "charger", "battery", "warranty",
            "diagnose", "replace", "support", "sorry",
        ],
        "action_items": [
            "Verify warranty status",
            "Attempt power reset with charger disconnected",
            "Schedule diagnostic appointment if unresolved",
        ],
    },
    # 2 ---------------------------------------------------------------
    {
        "ticket_id": "TKT-002",
        "customer_name": "Bob Martinez",
        "subject": "I was charged twice for my subscription",
        "body": (
            "My credit card statement shows two charges of $14.99 on the same day "
            "for my monthly subscription. Please fix this and refund the duplicate."
        ),
        "timestamp": "2025-06-01T10:30:00Z",
        "priority": "High",
        "department": "Billing",
        "response_keywords": [
            "refund", "duplicate", "charge", "billing", "sorry",
            "investigate", "statement", "credit",
        ],
        "action_items": [
            "Investigate duplicate charge in billing system",
            "Issue refund for the extra charge",
            "Confirm refund timeline with customer",
        ],
    },
    # 3 ---------------------------------------------------------------
    {
        "ticket_id": "TKT-003",
        "customer_name": "Carol Davis",
        "subject": "How do I reset my password?",
        "body": (
            "I forgot my password and can't log in. "
            "The 'Forgot Password' link doesn't seem to be sending me an email."
        ),
        "timestamp": "2025-06-01T11:00:00Z",
        "priority": "Low",
        "department": "General",
        "response_keywords": [
            "password", "reset", "email", "link", "spam",
            "account", "help", "login",
        ],
        "action_items": [
            "Verify customer email on file",
            "Trigger manual password reset link",
            "Check spam/junk folder",
        ],
    },
    # 4 ---------------------------------------------------------------
    {
        "ticket_id": "TKT-004",
        "customer_name": "David Kim",
        "subject": "Software crashes every time I open it",
        "body": (
            "After the latest update, the application crashes immediately on launch. "
            "I'm running Windows 11 and have tried reinstalling."
        ),
        "timestamp": "2025-06-02T08:00:00Z",
        "priority": "Medium",
        "department": "Tech",
        "response_keywords": [
            "crash", "update", "reinstall", "logs", "compatibility",
            "fix", "version", "troubleshoot", "sorry",
        ],
        "action_items": [
            "Collect crash logs from customer",
            "Check known issues with latest update",
            "Escalate to engineering if widespread",
        ],
    },
    # 5 ---------------------------------------------------------------
    {
        "ticket_id": "TKT-005",
        "customer_name": "Emma Wilson",
        "subject": "I want a refund for my purchase",
        "body": (
            "I bought the Pro plan three days ago but it doesn't have the features "
            "I expected. I would like a full refund per your 30-day guarantee."
        ),
        "timestamp": "2025-06-02T09:45:00Z",
        "priority": "Medium",
        "department": "Billing",
        "response_keywords": [
            "refund", "guarantee", "policy", "process", "sorry",
            "purchase", "plan", "confirm",
        ],
        "action_items": [
            "Verify purchase date within refund window",
            "Process refund per 30-day guarantee policy",
            "Offer alternative plan or downgrade",
        ],
    },
    # 6 ---------------------------------------------------------------
    {
        "ticket_id": "TKT-006",
        "customer_name": "Frank Nguyen",
        "subject": "Where is my order? It's been 3 weeks",
        "body": (
            "I placed order #ORD-8821 three weeks ago and still haven't received it. "
            "Tracking shows it's stuck in transit. This is unacceptable."
        ),
        "timestamp": "2025-06-02T14:20:00Z",
        "priority": "High",
        "department": "Escalation",
        "response_keywords": [
            "order", "shipping", "tracking", "sorry", "delayed",
            "investigate", "replacement", "expedite",
        ],
        "action_items": [
            "Contact shipping carrier for status update",
            "Offer replacement or expedited reshipping",
            "Escalate to logistics team",
        ],
    },
    # 7 ---------------------------------------------------------------
    {
        "ticket_id": "TKT-007",
        "customer_name": "Grace Lee",
        "subject": "Your website is showing a 500 error",
        "body": (
            "Every page on your website shows a 500 Internal Server Error. "
            "I can't access my account, make purchases, or even view products."
        ),
        "timestamp": "2025-06-03T07:10:00Z",
        "priority": "High",
        "department": "Tech",
        "response_keywords": [
            "error", "500", "server", "team", "investigating",
            "fix", "sorry", "status", "update",
        ],
        "action_items": [
            "Notify engineering / on-call team immediately",
            "Check server status dashboard",
            "Provide customer with status page link",
        ],
    },
    # 8 ---------------------------------------------------------------
    {
        "ticket_id": "TKT-008",
        "customer_name": "Henry Patel",
        "subject": "I need to update my billing address",
        "body": (
            "I recently moved and need to update my billing address. "
            "Where can I do that in my account settings?"
        ),
        "timestamp": "2025-06-03T10:00:00Z",
        "priority": "Low",
        "department": "Billing",
        "response_keywords": [
            "address", "update", "billing", "settings", "account",
            "profile", "navigate",
        ],
        "action_items": [
            "Provide step-by-step instructions for address update",
            "Offer to update address on behalf of customer if needed",
        ],
    },
    # 9 ---------------------------------------------------------------
    {
        "ticket_id": "TKT-009",
        "customer_name": "Isabella Chen",
        "subject": "How do I cancel my subscription?",
        "body": (
            "I'd like to cancel my monthly subscription. "
            "I couldn't find the cancellation option in my account dashboard."
        ),
        "timestamp": "2025-06-03T11:30:00Z",
        "priority": "Low",
        "department": "General",
        "response_keywords": [
            "cancel", "subscription", "account", "settings",
            "confirm", "help", "sorry",
        ],
        "action_items": [
            "Provide cancellation instructions",
            "Confirm if customer wants immediate or end-of-cycle cancellation",
            "Offer retention incentive if applicable",
        ],
    },
    # 10 --------------------------------------------------------------
    {
        "ticket_id": "TKT-010",
        "customer_name": "James Brown",
        "subject": "I received the wrong item",
        "body": (
            "I ordered a blue XL t-shirt but received a red medium instead. "
            "Order number is #ORD-5590. I need the correct item sent ASAP."
        ),
        "timestamp": "2025-06-04T08:45:00Z",
        "priority": "High",
        "department": "General",
        "response_keywords": [
            "wrong", "item", "order", "sorry", "replace",
            "correct", "return", "shipping", "label",
        ],
        "action_items": [
            "Send prepaid return label for wrong item",
            "Ship correct item (blue XL t-shirt) immediately",
            "Investigate fulfillment error",
        ],
    },
    # 11 --------------------------------------------------------------
    {
        "ticket_id": "TKT-011",
        "customer_name": "Karen Smith",
        "subject": "My account was hacked and someone made purchases",
        "body": (
            "I noticed several unauthorized purchases on my account totaling $347. "
            "I did NOT make these. Someone has accessed my account without permission."
        ),
        "timestamp": "2025-06-04T09:00:00Z",
        "priority": "High",
        "department": "Escalation",
        "response_keywords": [
            "unauthorized", "security", "account", "password", "sorry",
            "investigate", "freeze", "refund", "protect",
        ],
        "action_items": [
            "Immediately freeze/lock customer account",
            "Initiate fraud investigation",
            "Refund unauthorized purchases",
            "Force password reset and enable 2FA",
        ],
    },
    # 12 --------------------------------------------------------------
    {
        "ticket_id": "TKT-012",
        "customer_name": "Liam O'Brien",
        "subject": "The mobile app keeps freezing on Android",
        "body": (
            "The app freezes every few minutes on my Samsung Galaxy S24 running "
            "Android 15. I've cleared the cache and reinstalled but it still happens."
        ),
        "timestamp": "2025-06-04T12:00:00Z",
        "priority": "Medium",
        "department": "Tech",
        "response_keywords": [
            "app", "freeze", "android", "update", "version",
            "logs", "fix", "sorry", "team",
        ],
        "action_items": [
            "Collect device info and app version",
            "Ask customer to submit bug report from app",
            "Check known issues for Android 15 compatibility",
        ],
    },
    # 13 --------------------------------------------------------------
    {
        "ticket_id": "TKT-013",
        "customer_name": "Mia Taylor",
        "subject": "I'd like to upgrade my plan",
        "body": (
            "I'm currently on the Basic plan and want to switch to Premium. "
            "Will I be charged a prorated amount for this billing cycle?"
        ),
        "timestamp": "2025-06-05T10:15:00Z",
        "priority": "Low",
        "department": "Billing",
        "response_keywords": [
            "upgrade", "plan", "premium", "prorated", "billing",
            "charge", "confirm", "features",
        ],
        "action_items": [
            "Explain prorated billing for mid-cycle upgrade",
            "Provide upgrade instructions or process upgrade",
            "Highlight new Premium features",
        ],
    },
    # 14 --------------------------------------------------------------
    {
        "ticket_id": "TKT-014",
        "customer_name": "Noah Garcia",
        "subject": "Your support agent was rude to me",
        "body": (
            "I called your support line about a billing issue and the agent "
            "was dismissive, interrupted me, and hung up. Ticket ref: CALL-2201. "
            "This is completely unacceptable."
        ),
        "timestamp": "2025-06-05T14:30:00Z",
        "priority": "High",
        "department": "Escalation",
        "response_keywords": [
            "sorry", "apologize", "experience", "review", "feedback",
            "unacceptable", "manager", "investigate",
        ],
        "action_items": [
            "Apologize sincerely for poor experience",
            "Review call recording CALL-2201",
            "Escalate to support team manager",
            "Follow up with customer after review",
        ],
    },
    # 15 --------------------------------------------------------------
    {
        "ticket_id": "TKT-015",
        "customer_name": "Olivia White",
        "subject": "How do I export my data?",
        "body": (
            "I need to export all my account data (purchase history, profile info) "
            "as a CSV file. Is there a way to do this from the dashboard?"
        ),
        "timestamp": "2025-06-05T16:00:00Z",
        "priority": "Low",
        "department": "Tech",
        "response_keywords": [
            "export", "data", "csv", "dashboard", "settings",
            "download", "account", "guide",
        ],
        "action_items": [
            "Provide data export instructions",
            "Link to relevant help article / documentation",
        ],
    },
]

# Quick lookup by ticket_id
_TICKET_GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    t["ticket_id"]: t for t in TICKET_DATABASE
}

# Task descriptions
TASK_DESCRIPTIONS = {
    1: (
        "EASY – Priority Classification: Classify each ticket as Low, Medium, "
        "or High priority."
    ),
    2: (
        "MEDIUM – Department Routing: Classify priority AND route to the "
        "correct department (Tech / Billing / General / Escalation)."
    ),
    3: (
        "HARD – Full Response: Classify priority, route department, write a "
        "helpful first response, and extract action items."
    ),
}


# ===================================================================
# Grading helpers
# ===================================================================

PRIORITY_LEVELS = {"Low": 0, "Medium": 1, "High": 2}
VALID_DEPARTMENTS = {"Tech", "Billing", "General", "Escalation"}


def _grade_priority(predicted: str, expected: str) -> float:
    """1.0 exact, 0.5 adjacent, 0.0 off-by-two or invalid."""
    pred_val = PRIORITY_LEVELS.get(predicted)
    exp_val = PRIORITY_LEVELS.get(expected)
    if pred_val is None or exp_val is None:
        return 0.0
    diff = abs(pred_val - exp_val)
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.5
    return 0.0


def _grade_department(predicted: str, expected: str) -> float:
    """1.0 exact match, 0.0 otherwise."""
    if predicted not in VALID_DEPARTMENTS:
        return 0.0
    return 1.0 if predicted == expected else 0.0


def _grade_response(response: str, expected_keywords: List[str]) -> float:
    """Fraction of expected keywords found (case-insensitive) in response."""
    if not expected_keywords or not response:
        return 0.0
    response_lower = response.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    return round(hits / len(expected_keywords), 4)


def _grade_action_items(
    predicted: List[str], expected: List[str]
) -> float:
    """
    For each expected action item, check if *any* predicted item has
    meaningful word overlap (≥50 % of words). Returns fraction matched.
    """
    if not expected:
        return 1.0  # nothing expected → full marks
    if not predicted:
        return 0.0

    def _word_set(text: str) -> set:
        return set(text.lower().split())

    matched = 0
    for exp in expected:
        exp_words = _word_set(exp)
        for pred in predicted:
            pred_words = _word_set(pred)
            if not exp_words:
                continue
            overlap = len(exp_words & pred_words) / len(exp_words)
            if overlap >= 0.40:
                matched += 1
                break
    return round(matched / len(expected), 4)


def _compute_penalty(
    predicted_priority: str,
    expected_priority: str,
    predicted_department: str,
    expected_department: str,
) -> float:
    """
    Small penalties for egregiously wrong actions:
      -0.10  if High ↔ Low priority confusion
      -0.05  if routed to a clearly wrong department category
    Penalty is clamped so total score never goes below 0.
    """
    penalty = 0.0
    p_val = PRIORITY_LEVELS.get(predicted_priority)
    e_val = PRIORITY_LEVELS.get(expected_priority)
    if p_val is not None and e_val is not None and abs(p_val - e_val) == 2:
        penalty -= 0.10
    if (
        predicted_department in VALID_DEPARTMENTS
        and expected_department in VALID_DEPARTMENTS
        and predicted_department != expected_department
    ):
        # Extra penalty for cross-category mistakes (tech ↔ billing)
        cross = {
            frozenset({"Tech", "Billing"}),
            frozenset({"General", "Escalation"}),
        }
        if frozenset({predicted_department, expected_department}) in cross:
            penalty -= 0.05
    return penalty


def grade_ticket(
    action: SupportTicketRouterAction,
    ground_truth: Dict[str, Any],
    task_id: int,
) -> TicketScore:
    """
    Deterministic grading of a single ticket action.

    Task weights
    ------------
    Task 1: priority 1.0
    Task 2: priority 0.50, department 0.50
    Task 3: priority 0.25, department 0.25, response 0.30, action_items 0.20
    """
    pri_score = _grade_priority(action.priority, ground_truth["priority"])
    dept_score = _grade_department(action.department, ground_truth["department"])
    resp_score = _grade_response(
        action.response, ground_truth.get("response_keywords", [])
    )
    ai_score = _grade_action_items(
        action.action_items, ground_truth.get("action_items", [])
    )
    penalty = _compute_penalty(
        action.priority,
        ground_truth["priority"],
        action.department,
        ground_truth["department"],
    )

    # Weighted score based on task
    if task_id == 1:
        weighted = pri_score
    elif task_id == 2:
        weighted = 0.50 * pri_score + 0.50 * dept_score
    else:
        weighted = (
            0.25 * pri_score
            + 0.25 * dept_score
            + 0.30 * resp_score
            + 0.20 * ai_score
        )

    weighted = round(max(0.0, min(1.0, weighted + penalty)), 4)

    return TicketScore(
        ticket_id=action.ticket_id,
        priority_score=pri_score,
        department_score=dept_score,
        response_score=resp_score,
        action_items_score=ai_score,
        penalty=penalty,
        task_weighted_score=weighted,
    )


# ===================================================================
# Environment
# ===================================================================

class SupportTicketRouterEnvironment(Environment):
    """
    OpenEnv-compatible environment for the Customer Support Ticket Router.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id: int = 1
        self._current_tickets: List[Dict[str, Any]] = []
        self._processed: Dict[str, TicketScore] = {}
        self._rng = random.Random()  # unseeded → different tickets each run

    # ------------------------------------------------------------------
    # Public helpers (used by app.py endpoints)
    # ------------------------------------------------------------------

    def set_task(self, task_id: int) -> None:
        """Set the active task (1, 2, or 3). Call before reset()."""
        if task_id not in (1, 2, 3):
            raise ValueError("task_id must be 1, 2, or 3")
        self._task_id = task_id

    def get_episode_scores(self) -> Dict[str, Any]:
        """Return aggregated scores for the current episode."""
        if not self._processed:
            return {"episode_id": self._state.episode_id, "scores": [], "mean_score": 0.0}
        scores = list(self._processed.values())
        mean = round(
            sum(s.task_weighted_score for s in scores) / len(scores), 4
        )
        return {
            "episode_id": self._state.episode_id,
            "task_id": self._task_id,
            "scores": [s.model_dump() for s in scores],
            "mean_score": mean,
        }

    def get_current_tickets(self) -> List[Dict[str, Any]]:
        """Return raw ticket dicts (with ground truth) for current episode."""
        return deepcopy(self._current_tickets)

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(self) -> SupportTicketRouterObservation:
        """Sample 1-3 tickets and start a new episode."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._processed = {}

        num_tickets = self._rng.randint(1, 3)
        self._current_tickets = self._rng.sample(TICKET_DATABASE, k=num_tickets)

        tickets = [
            TicketData(
                ticket_id=t["ticket_id"],
                customer_name=t["customer_name"],
                subject=t["subject"],
                body=t["body"],
                timestamp=t["timestamp"],
            )
            for t in self._current_tickets
        ]

        return SupportTicketRouterObservation(
            tickets=tickets,
            task_id=self._task_id,
            task_description=TASK_DESCRIPTIONS[self._task_id],
            tickets_remaining=len(tickets),
            last_score=None,
            done=False,
            reward=0.0,
            metadata={
                "episode_id": self._state.episode_id,
                "num_tickets": num_tickets,
                "ticket_ids": [t["ticket_id"] for t in self._current_tickets],
            },
        )

    def step(
        self, action: SupportTicketRouterAction
    ) -> SupportTicketRouterObservation:
        """
        Process the agent's action for one ticket.

        Returns observation with grading feedback. When all tickets in the
        episode are processed, `done=True` and `reward` is the mean
        task-weighted score across tickets.
        """
        self._state.step_count += 1

        # Look up ground truth
        gt = _TICKET_GROUND_TRUTH.get(action.ticket_id)
        if gt is None:
            # Unknown ticket id → zero score, not done
            bad_score = TicketScore(ticket_id=action.ticket_id)
            remaining = len(self._current_tickets) - len(self._processed)
            return SupportTicketRouterObservation(
                tickets=[],
                task_id=self._task_id,
                task_description=TASK_DESCRIPTIONS[self._task_id],
                tickets_remaining=remaining,
                last_score=bad_score,
                done=False,
                reward=0.0,
                metadata={"error": f"Unknown ticket_id: {action.ticket_id}"},
            )

        # Grade
        score = grade_ticket(action, gt, self._task_id)
        self._processed[action.ticket_id] = score

        remaining = len(self._current_tickets) - len(self._processed)
        is_done = remaining <= 0

        # Episode reward = mean weighted score when done, else running mean
        all_scores = list(self._processed.values())
        mean_reward = round(
            sum(s.task_weighted_score for s in all_scores) / len(all_scores), 4
        )

        return SupportTicketRouterObservation(
            tickets=[],  # tickets only sent on reset
            task_id=self._task_id,
            task_description=TASK_DESCRIPTIONS[self._task_id],
            tickets_remaining=max(remaining, 0),
            last_score=score,
            done=is_done,
            reward=mean_reward,
            metadata={
                "step": self._state.step_count,
                "episode_id": self._state.episode_id,
                "tickets_processed": len(self._processed),
                "running_mean_score": mean_reward,
            },
        )

    @property
    def state(self) -> State:
        return self._state
