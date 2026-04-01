#!/usr/bin/env python3
"""
Baseline inference client for the Customer Support Ticket Router.

Uses OpenAI or Google Gemini API to play all three tasks against the 
environment server and prints per-task and per-episode scores.

Usage:
    # With .env file (recommended)
    python client.py --provider gemini --model gemini-2.0-flash --episodes 5
    
    # With environment variable
    set OPENAI_API_KEY="sk-..."
    python client.py --provider openai --model gpt-4o-mini --episodes 5

Requirements:
    pip install openai google-generativeai requests python-dotenv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import requests

# Load .env file if exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional


# ------------------------------------------------------------------
# LLM Client wrappers for different providers
# ------------------------------------------------------------------

class OpenAIClient:
    """Wrapper for OpenAI API."""

    def __init__(self, api_key: str, model: str):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            print("ERROR: `openai` package required. pip install openai")
            sys.exit(1)
        self.model = model

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class GeminiClient:
    """Wrapper for Google Gemini API."""

    def __init__(self, api_key: str, model: str):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
        except ImportError:
            print(
                "ERROR: `google-generativeai` package required. pip install google-generativeai")
            sys.exit(1)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        # Convert OpenAI-style messages to Gemini format
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += msg["content"]

        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        )
        return response.text


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def env_reset(base: str, task_id: int) -> Dict[str, Any]:
    """Call POST /reset on the environment server."""
    resp = requests.post(f"{base}/reset")
    resp.raise_for_status()
    data = resp.json()
    # Response may be wrapped in {"observation": ...} or bare
    return data.get("observation", data)


def env_step(base: str, action: Dict[str, Any]) -> Dict[str, Any]:
    """Call POST /step on the environment server."""
    resp = requests.post(f"{base}/step", json=action)
    resp.raise_for_status()
    data = resp.json()
    # Response may be wrapped in {"observation": ...} or bare
    return data.get("observation", data)


def get_tasks(base: str) -> List[Dict[str, Any]]:
    """Call GET /tasks on the environment server."""
    resp = requests.get(f"{base}/tasks")
    resp.raise_for_status()
    return resp.json()


def build_prompt(ticket: Dict[str, Any], task_id: int) -> str:
    """Construct the LLM prompt for a single ticket."""
    header = (
        f"Support Ticket ID: {ticket['ticket_id']}\n"
        f"Customer: {ticket['customer_name']}\n"
        f"Subject: {ticket['subject']}\n"
        f"Body: {ticket['body']}\n"
        f"Timestamp: {ticket['timestamp']}\n\n"
    )

    if task_id == 1:
        header += (
            "Classify the priority of this ticket as exactly one of: "
            "Low, Medium, or High.\n"
            'Respond with JSON only: {"priority": "..."}\n'
        )
    elif task_id == 2:
        header += (
            "Classify the priority (Low / Medium / High) and route to a "
            "department (Tech / Billing / General / Escalation).\n"
            'Respond with JSON only: {"priority": "...", "department": "..."}\n'
        )
    else:
        header += (
            "Classify the priority (Low / Medium / High), route to a department "
            "(Tech / Billing / General / Escalation), write a helpful first "
            "response to the customer, and list action items for the support team.\n"
            "Respond with JSON only:\n"
            "{\n"
            '  "priority": "...",\n'
            '  "department": "...",\n'
            '  "response": "Your helpful response here...",\n'
            '  "action_items": ["item 1", "item 2"]\n'
            "}\n"
        )
    return header


def parse_llm_json(raw: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from LLM output."""
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_task(
    base_url: str,
    llm_client,
    model: str,
    task_id: int,
    num_episodes: int,
) -> Dict[str, Any]:
    """Run multiple episodes for a single task and return results."""
    print(f"\n{'='*60}")
    print(f"  TASK {task_id}")
    print(f"{'='*60}")

    episode_scores: List[float] = []

    for ep in range(num_episodes):
        # Reset environment
        obs = env_reset(base_url, task_id)
        tickets = obs.get("tickets", [])
        if not tickets:
            print(f"  Episode {ep+1}: No tickets received, skipping.")
            continue

        print(f"\n  Episode {ep+1}: {len(tickets)} ticket(s)")

        ticket_scores: List[float] = []

        for ticket in tickets:
            prompt = build_prompt(ticket, task_id)

            # Call LLM
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a customer support triage AI. "
                        "Respond ONLY with valid JSON. No markdown fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            raw = llm_client.chat(messages, temperature=0.0, max_tokens=1024)
            parsed = parse_llm_json(raw)

            # Build action
            action = {
                "ticket_id": ticket["ticket_id"],
                "priority": parsed.get("priority", "Medium"),
                "department": parsed.get("department", "General"),
                "response": parsed.get("response", ""),
                "action_items": parsed.get("action_items", []),
            }

            # Step
            step_obs = env_step(base_url, action)
            last = step_obs.get("last_score", {})
            tw = last.get("task_weighted_score", 0.0)
            ticket_scores.append(tw)

            print(
                f"    {ticket['ticket_id']}: "
                f"pri={action['priority']:>6s}  "
                f"dept={action['department']:>11s}  "
                f"score={tw:.3f}  "
                f"(pri={last.get('priority_score', 0):.1f} "
                f"dept={last.get('department_score', 0):.1f} "
                f"resp={last.get('response_score', 0):.2f} "
                f"ai={last.get('action_items_score', 0):.2f})"
            )

        ep_mean = sum(ticket_scores) / \
            len(ticket_scores) if ticket_scores else 0.0
        episode_scores.append(ep_mean)
        print(f"    → Episode mean score: {ep_mean:.4f}")

    overall = (
        sum(episode_scores) / len(episode_scores) if episode_scores else 0.0
    )
    print(f"\n  Task {task_id} overall mean: {overall:.4f}")
    return {"task_id": task_id, "episode_scores": episode_scores, "mean": overall}


def main():
    parser = argparse.ArgumentParser(
        description="Baseline client for Customer Support Ticket Router"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Environment server URL",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default="gemini",
        help="LLM provider to use (default: gemini)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Model to use (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes per task",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Which task(s) to run (1, 2, 3)",
    )
    args = parser.parse_args()

    # Get API key based on provider
    if args.provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: Set GEMINI_API_KEY in .env file or environment.")
            sys.exit(1)
        llm_client = GeminiClient(api_key, args.model)
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: Set OPENAI_API_KEY in .env file or environment.")
            sys.exit(1)
        llm_client = OpenAIClient(api_key, args.model)

    # Print available tasks
    print("Fetching task definitions from server...")
    try:
        tasks_info = get_tasks(args.base_url)
        for t in tasks_info:
            print(
                f"  Task {t['task_id']}: {t['name']} — {t['description'][:80]}...")
    except Exception as e:
        print(f"  (Could not fetch /tasks: {e})")

    # Run each requested task
    results = []
    for tid in args.tasks:
        result = run_task(args.base_url, llm_client,
                          args.model, tid, args.episodes)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model}")
    print(f"  Episodes per task: {args.episodes}")
    for r in results:
        print(
            f"  Task {r['task_id']}: mean={r['mean']:.4f}  per-episode={r['episode_scores']}")
    overall_all = sum(r["mean"] for r in results) / \
        len(results) if results else 0.0
    print(f"  Overall mean across tasks: {overall_all:.4f}")
    print()


if __name__ == "__main__":
    main()
