#!/usr/bin/env python3
"""
Manual test script for the Customer Support Ticket Router.
No API key needed - you make the decisions!
"""

import requests

BASE = "http://localhost:8000"

print("=" * 60)
print("  MANUAL TEST - Customer Support Ticket Router")
print("=" * 60)

# Get tasks
print("\nFetching tasks...")
tasks = requests.get(f"{BASE}/tasks").json()
for t in tasks:
    print(f"  Task {t['task_id']}: {t['name']}")

# Reset
print("\nResetting environment...")
resp = requests.post(f"{BASE}/reset")
obs = resp.json().get("observation", resp.json())
tickets = obs.get("tickets", [])
print(f"Received {len(tickets)} ticket(s)\n")

# Process each ticket
for ticket in tickets:
    print("-" * 60)
    print(f"TICKET: {ticket['ticket_id']}")
    print(f"Customer: {ticket['customer_name']}")
    print(f"Subject: {ticket['subject']}")
    print(f"Body: {ticket['body'][:200]}...")
    print()
    
    # Get user input
    print("Your decision:")
    priority = input("  Priority (Low/Medium/High): ").strip() or "Medium"
    department = input("  Department (Tech/Billing/General/Escalation): ").strip() or "General"
    response_text = input("  Response to customer: ").strip() or "Thank you for contacting us."
    action_items_input = input("  Action items (comma-separated): ").strip()
    action_items = [x.strip() for x in action_items_input.split(",") if x.strip()]
    
    # Submit
    action = {
        "ticket_id": ticket["ticket_id"],
        "priority": priority,
        "department": department,
        "response": response_text,
        "action_items": action_items,
    }
    
    step_resp = requests.post(f"{BASE}/step", json=action)
    step_obs = step_resp.json().get("observation", step_resp.json())
    score = step_obs.get("last_score", {})
    
    print(f"\n  Score: {score.get('task_weighted_score', 0):.3f}")
    print(f"    Priority: {score.get('priority_score', 0):.1f}")
    print(f"    Department: {score.get('department_score', 0):.1f}")
    print(f"    Response: {score.get('response_score', 0):.2f}")
    print(f"    Action Items: {score.get('action_items_score', 0):.2f}")
    print()

# Final summary
print("=" * 60)
print("  EPISODE COMPLETE")
print("=" * 60)
print(f"  Final reward: {step_obs.get('reward', 0):.3f}")
print()
