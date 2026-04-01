# Customer Support Ticket Router

**Python 3.10+** | **FastAPI** | **OpenEnv** | **License: MIT**

An OpenEnv environment that simulates a customer support inbox. An AI agent reads realistic support tickets and must triage them — classifying priority, routing to the correct department, drafting a helpful response, and extracting action items. A deterministic grader scores every action from **0.0 to 1.0**.

Built for the **Meta PyTorch Hackathon**.

---

## Try It Yourself (No AI API Key Needed)

The easiest way to explore this project is to **be the AI yourself**. You'll read support tickets and make triage decisions — the system scores you instantly. No setup, no API keys, no terminal experience required.

---

### Step 1 — Install Python

If you don't already have Python installed:

1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Download the latest version (**3.10 or higher**)
3. Run the installer
4. ⚠️ **Important:** Check the box that says **"Add Python to PATH"** — this is required
5. Click **Install**

To verify it worked, open a terminal (see Step 3) and type:

```
python --version
```

You should see something like `Python 3.12.x`. If you see an error, revisit the installer and make sure "Add Python to PATH" was checked.

---

### Step 2 — Download the Project

1. Go to the [GitHub repository page](https://github.com/tanushgargjanuary-bot/hara-baingan-ticket-router)
2. Click the green **"⬇ Code"** button near the top-right
3. Click **"Download ZIP"**
4. Find the downloaded `.zip` file (usually in your **Downloads** folder)
5. **Unzip it:**
   - **Windows:** Right-click the `.zip` file → **Extract All** → Choose a location → Click **Extract**
   - **Mac:** Double-click the `.zip` file — it will unzip automatically
   - **Linux:** Right-click → **Extract Here**

You should now have a folder called `hara-baingan-ticket-router-main`.

---

### Step 3 — Start the Server

Choose your operating system:

#### Windows
Open the unzipped folder and **double-click `run_server.bat`**.

#### Mac
Open the unzipped folder and **double-click `run_server.command`**.

> **If Mac says the file "can't be opened":**
> 1. Right-click `run_server.command`
> 2. Click **Open With → Terminal**
> 3. If prompted, click **Open** to confirm

#### Linux
1. Open a terminal
2. Navigate to the project folder:
   ```
   cd ~/Downloads/hara-baingan-ticket-router-main
   ```
3. Run:
   ```
   chmod +x run_server.command
   ./run_server.command
   ```

---

A terminal window will open, install the required packages automatically, and launch your browser to **http://localhost:8000**. You'll see the API info page — this means the server is running. (It will look like raw data, not a pretty website — that's normal!)

---

### Step 4 — Try the Manual Test

**Leave the server running** (don't close that terminal window), and open a **second terminal:**

- **Windows:** Press `Win + R`, type `cmd`, hit Enter
- **Mac:** Press `Cmd + Space`, type `Terminal`, hit Enter
- **Linux:** Press `Ctrl + Alt + T`

Now navigate to the project folder. For example, if you unzipped it into your Downloads folder:

```
cd ~/Downloads/hara-baingan-ticket-router-main
```

> 💡 **Windows users:** The command might look like this instead:
> ```
> cd C:\Users\YourName\Downloads\hara-baingan-ticket-router-main
> ```

Then run:

```
python test_manual.py
```

You'll see a support ticket and be asked to classify it. Type your answers and get scored instantly. **No AI account or API key required.**

---

### What You'll See

```
============================================================
  MANUAL TEST - Customer Support Ticket Router
============================================================

Fetching tasks...
  Task 1: Priority Classification
  Task 2: Department Routing
  Task 3: Full Response

Resetting environment...
Received 2 ticket(s)

------------------------------------------------------------
TICKET: TKT-003
Customer: Carol Davis
Subject: How do I reset my password?
Body: I forgot my password and can't log in...

Your decision:
  Priority (low/Medium/High): high
  Department (Tech/Billing/General/Escalation): General
  Response to customer: Sorry about that, I've sent a reset link.
  Action items (comma-separated): Check email, send reset link

  Score: 0.650
    Priority: 0.5
    Department: 1.0
    Response: 0.38
    Action Items: 0.67
```

---

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────┐
│  /reset     │────▶│ 1-3 tickets  │────▶│ Agent processes  │────▶│ /step ×N │
│  (new round)│     │ returned     │     │ each ticket      │     │ (grade)  │
└─────────────┘     └──────────────┘     └──────────────────┘     └──────────┘
                                                                        │
                                                                        ▼
                                                               ┌─────────────────┐
                                                               │  Score: 0.0-1.0 │
                                                               │  per ticket     │
                                                               └─────────────────┘
```

1. **Reset** — the environment returns 1–3 random tickets from a pool of 15
2. **Triage** — the agent (or you!) reads each ticket and submits a structured action
3. **Grade** — each action is scored against the correct answers using consistent rules (same answers always produce the same score — no randomness)
4. **Done** — when all tickets are processed, your final score is the average across all tickets

---

## Tasks

Three difficulty tiers, each building on the last:

| Task | Difficulty | What You Do | Graded Fields |
|------|-----------|-------------|---------------|
| 1 | Easy | Classify priority | priority |
| 2 | Medium | Classify priority + route to department | priority, department |
| 3 | Hard | Priority + department + response + action items | All four fields |

---

## Scoring

You don't have to be perfect — the system gives **partial credit**:

| Field | Exact Match | Partial Credit |
|-------|-------------|----------------|
| **Priority** | 1.0 | **0.5** if you're one level off (e.g., you said Medium but the answer was Low) |
| **Department** | 1.0 | None — it's either right or wrong |
| **Response** | — | Based on how many key phrases from the ideal response appear in yours |
| **Action Items** | — | If 40% or more of the words match the expected item, it counts |

**Penalties** (to discourage random guessing):

- **−0.10** for getting priority completely wrong (e.g., saying High when it should be Low)
- **−0.05** for routing to the wrong category entirely (e.g., Tech when it should be Billing)

---

## Example

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

## Developer Quick Start

> 🛠️ **This section is for developers comfortable with the terminal.** If you followed Steps 1–4 above, you can skip this.

```bash
# Install dependencies
pip install -r server/requirements.txt

# Start the server
python -m server.app --port 8000

# Or use the one-click launcher (auto-opens browser)
python run_server.py
```

Visit **http://localhost:8000** to verify it's running.

---

### Running the Baseline Agent

Test how well an AI agent performs. You'll need an API key from [OpenAI](https://platform.openai.com/) or [Google AI](https://aistudio.google.com/).

```bash
# Set up your API key
cp .env.example .env
# Edit .env and 
