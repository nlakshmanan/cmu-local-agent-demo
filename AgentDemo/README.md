# Teacher's Assistant — Local Agent Demo

A small AI agent that runs entirely on your laptop. It plays the part of a
professor's assistant: it can pull up students, crunch class stats, draw
charts, and remember things you tell it between sessions. No API keys, no
cloud, no cost.

You'll build your capstone off this, so get it running before class.

## Setup — do this BEFORE class

The model download is about 3.5 GB. Classroom wifi will not save you.

**1. Install [Ollama](https://ollama.com/download)**, then pull the two models:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

**2. Clone this repo and make an environment:**

```bash
git clone <this-repo-url>
cd AgentDemo
conda create -n agentdemo python=3.11 -y
conda activate agentdemo
pip install -r requirements.txt
```

(venv works fine too if you don't use conda.)

**3. Check everything:**

```bash
python setup_check.py
```

All green means you're ready. Anything red prints the exact command to fix it.

**4. Run it:**

```bash
python app.py
```

Your browser opens to `http://127.0.0.1:7860`. Ask it "How is Sam Rivera
doing?" and watch the Trace tab while it answers.

**Slow machine?** Open `config.py` and switch `MODEL` to `gemma3:1b`
(pull it first). Everything still works, the answers are just less sharp.

## What each file is

| File | What it is |
|---|---|
| `agent.py` | The brain. A ~90 line loop: "do I need a tool? call it. now answer." Read this one first. |
| `course_server.py` | The hands. A separate program with the gradebook tools, spoken to over MCP. The agent can think all it wants; only tools can actually touch data. |
| `course_data.json` | The gradebook. The model has never seen this file. Tools are the only way in. |
| `memory.py` | The diary. Durable facts get written to disk after each turn and looked up by meaning later. This is long-term memory. |
| `skills/` | The recipe book. Step-by-step procedures (like how to write a check-in email) that load into the prompt only when needed, so they don't eat the context window while idle. |
| `mcp_client.py` | The phone line between the agent and the tool server. Plumbing, safe to skim. |
| `app.py` | The face. The chat window plus side panels that show exactly what the agent can see at any moment. |
| `config.py` | The knobs. Model choice, memory sizes, thresholds. Change things here first. |

Short-term memory doesn't get its own file because it doesn't need one: it's
a plain Python list of messages in `agent.py`, and when it fills up, old
messages fall off and are simply gone. That's a context window.

## Six things to try

1. `What is 4820 * 1834?` — the model hands the math to a tool, because
   models are confidently bad at arithmetic.
2. `How is Sam Rivera doing?` — that data isn't in the model. Watch the
   Trace tab: it fetched it over MCP and then noticed the trend itself.
3. `Who are all of my students?` then `Show me stats for each one.` — both
   land on the roster tool. This one exists because of a real bug: before it
   did, the agent had no tool that fit and confidently made up eight students'
   averages. When an agent fabricates, the fix is usually a missing tool, not
   a better prompt.
3. `Show me a chart of the class.` — a tool draws the PNG, the model
   explains it. Each half doing the job the other can't.
4. Tell it `Priya has an extended-time accommodation for exams.` Hit
   **Reset conversation**, then ask `Anything I should keep in mind for the
   final?` It remembers — from disk, not from the chat. Now flip retrieval
   to `keyword` and ask again: nothing. That difference is why embeddings
   exist.
5. Hit **Fill context window**, then ask about something from earlier in the
   chat. Gone. That's short-term memory overflowing in real time.

## Making it yours

Search the code for `---- MODIFY HERE ----` comments — those mark the knobs
worth experimenting with. Adding a tool is one function in
`course_server.py`; the agent discovers it at startup on its own.

## If something breaks

| Symptom | Fix |
|---|---|
| `connection refused` at startup | Ollama isn't running. Start the app, or run `ollama serve`. |
| `model not found` | `ollama pull gemma3:4b` |
| Replies crawl | Normal on CPU. Use a smaller `MODEL` in `config.py`. |
| It answered with numbers but called no tool | It made them up. Small models do this — open the Trace tab and catch it in the act. This is why evals exist. |
| Memory panel stays empty | Only durable facts get saved, not questions. Tell it something worth writing down. |
| Want a clean slate | Hit **Wipe long-term memory**, or delete `memories.json`. |
