"""
app.py — The GUI. Run this file.

    python app.py

Layout:

    +--------------------------+-----------------------------------+
    |                          | [Context] [Memory] [Tools] [Trace] |
    |   chat with the agent    |                                   |
    |                          |   live panels showing what the    |
    |                          |   agent can actually SEE          |
    |  [ type here ]   [send]  |                                   |
    +--------------------------+-----------------------------------+

The panels are the point. The chat is just how you poke it.

Note the deliberate split: the chat window on the left keeps the FULL history
for you to read, while the Context panel on the right shows only the messages
still being sent to the model. When those two disagree, you are looking at the
context window limit with your own eyes.
"""

import threading

import gradio as gr
import ollama

import config
import memory
import skills_loader
from agent import run_turn
from mcp_client import MCPClient
import map_view          # builds the interactive site map from the agent's output

# ---------------------------------------------------------------------------
# STARTUP
# ---------------------------------------------------------------------------
print("Connecting to MCP server...")
MCP = MCPClient(config.MCP_SERVER_SCRIPT)
MCP.connect()
print(f"  connected. Tools discovered: {[t.name for t in MCP.tools]}")


def _warm_up():
    """Load the model into RAM now, so the first question isn't 15s slower."""
    try:
        ollama.chat(
            model=config.MODEL,
            messages=[{"role": "user", "content": "hi"}],
            keep_alive=config.KEEP_ALIVE,
        )
        print(f"  {config.MODEL} warmed up and resident.")
    except Exception as e:
        print(f"  WARNING: could not reach Ollama ({e}). Run `ollama serve`.")


threading.Thread(target=_warm_up, daemon=True).start()


# ---------------------------------------------------------------------------
# PANEL RENDERERS — turn state into readable markdown
# ---------------------------------------------------------------------------
def render_context(messages: list[dict]) -> str:
    """SHORT-TERM MEMORY: exactly what gets sent to the model."""
    limit = config.SHORT_TERM_MAX_TURNS * 2
    header = (
        f"### Short-term memory\n"
        f"`{len(messages)} / {limit}` messages in the window "
        f"(= {config.SHORT_TERM_MAX_TURNS} turns, set in `config.py`)\n\n"
    )
    if len(messages) >= limit:
        header += "> **FULL.** Every new message now pushes an old one out permanently.\n\n"
    if not messages:
        return header + "_empty — nothing has been said yet_"

    rows = []
    for m in messages:
        text = m["content"].replace("\n", " ")
        text = text[:90] + ("..." if len(text) > 90 else "")
        rows.append(f"| {m['role']} | {text} |")
    return header + "| role | content |\n|---|---|\n" + "\n".join(rows)


def render_memory() -> str:
    """LONG-TERM MEMORY: the contents of memories.json."""
    memories = memory.load()
    header = f"### Long-term memory\n`{len(memories)}` fact(s) in `{config.MEMORY_FILE}`\n\n"
    if not memories:
        return header + "_empty — tell the agent a priority the tools don't know (a constraint, a preference, a must-have)_"
    return header + "\n".join(f"- **#{m['id']}** {m['text']}" for m in memories)


def render_tools() -> str:
    """What the agent discovered at runtime — nothing here is hardcoded."""
    out = ["### Tools (discovered live over MCP)"]
    out.append(f"_Server: `{config.MCP_SERVER_SCRIPT}`, running as a separate process._\n")
    for tool in MCP.tools:
        params = ", ".join(tool.input_schema.get("properties", {}).keys()) or ""
        out.append(f"- **`{tool.name}({params})`** — {tool.description.strip().splitlines()[0]}")
    out.append("- **`load_skill(name)`** — _client-side, not from MCP_")

    out.append("\n### Skills (names + descriptions only)")
    out.append("_Full instructions load on demand. That's progressive disclosure._\n")
    for skill in skills_loader.list_skills():
        out.append(f"- **{skill['name']}** — {skill['description']}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# THE MAIN EVENT HANDLER
# ---------------------------------------------------------------------------
def on_send(user_text, chat, messages, retrieval_mode):
    """Runs one turn and streams every intermediate state into the GUI."""
    if not user_text.strip():
        yield user_text, chat, messages, "", render_context(messages), render_memory(), ""
        return

    chat = chat + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ""},
    ]
    trace_lines, stats = [], ""

    # run_turn is a generator; each event updates a different part of the UI.
    for kind, payload in run_turn(MCP, messages, user_text, retrieval_mode):
        if kind == "trace":
            trace_lines.append(payload)
        elif kind == "chart":
            # A tool rendered a PNG. Show the image itself in the chat, just
            # above the answer that's about to stream in. The LLM never sees
            # this image -- tools return artifacts, the model returns words.
            chat.insert(len(chat) - 1, {"role": "assistant", "content": gr.Image(payload)})
        elif kind == "token":
            chat[-1]["content"] += payload
        elif kind == "stats":
            stats = (
                f"**Prompt size:** {payload['prompt_tokens']} tokens &nbsp;|&nbsp; "
                f"**Skills loaded:** {payload['skills_loaded']}"
            )
        elif kind == "done":
            messages = payload

        yield (
            "",
            chat,
            messages,
            "### Trace\n" + "\n\n".join(trace_lines),
            render_context(messages),
            render_memory(),
            stats,
        )


# ---------------------------------------------------------------------------
# DEMO BUTTONS — so you can re-run a beat without retyping
# ---------------------------------------------------------------------------
def reset_conversation():
    """Clear short-term memory only. Long-term facts survive — that's the point."""
    return [], [], render_context([]), render_memory(), "", ""


def wipe_long_term():
    memory.wipe()
    return render_memory()


def fill_context(messages):
    """Stuff the window with filler so you can show overflow instantly."""
    filler = []
    for i in range(config.SHORT_TERM_MAX_TURNS):
        filler.append({"role": "user", "content": f"[filler question #{i + 1}]"})
        filler.append({"role": "assistant", "content": f"[filler answer #{i + 1}]"})
    messages = (messages + filler)[-config.SHORT_TERM_MAX_TURNS * 2 :]
    return messages, render_context(messages)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
# Look and feel: a slate + emerald theme with a banner header and centered
# layout, so the app reads as its own product rather than the stock template.
THEME = gr.themes.Soft(primary_hue="emerald", secondary_hue="teal", neutral_hue="slate")

CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
#app-header {
  background: linear-gradient(135deg, #0e2f29 0%, #103a52 100%);
  border: 1px solid #1f6152; border-radius: 14px;
  padding: 18px 22px; margin-bottom: 14px;
}
#app-header h1 { margin: 0 0 4px 0; font-size: 1.6rem; letter-spacing: .2px; color: #e9fff8; }
#app-header p  { margin: 0; color: #9fd8c9; font-size: .95rem; }

/* Fixed-height side panel so switching tabs never resizes the layout. */
#side-panel .tabitem, #side-panel [role="tabpanel"] {
  height: 560px; min-height: 560px; max-height: 560px;
  overflow-y: auto; border-radius: 12px;
}
footer { display: none !important; }
"""

with gr.Blocks(title="Data-Center Site Forecaster", theme=THEME, css=CUSTOM_CSS) as demo:
    gr.Markdown(
        "# Data-Center Site Forecaster\n"
        "An agent for data-center site selection.",
        elem_id="app-header",
    )

    # Short-term memory really is just a Python list living in this State.
    messages_state = gr.State([])

    with gr.Row():
        # ---------------- LEFT: chat ----------------
        with gr.Column(scale=5):
            # Gradio 6 chat messages are {"role": ..., "content": ...} dicts --
            # the same shape the Ollama API uses, so no conversion needed.
            chatbot = gr.Chatbot(height=460, show_label=False)
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask about a candidate site, request a forecast or chart, or share a priority only you'd know...",
                    show_label=False,
                    scale=8,
                    autofocus=True,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            stats_md = gr.Markdown("")

            with gr.Row():
                retrieval_mode = gr.Radio(
                    ["semantic", "keyword"],
                    value="semantic",
                    label="Memory retrieval strategy",
                    info="Ask the same question with each. Watch the Trace panel.",
                    scale=2,
                )
                with gr.Column(scale=1):
                    reset_btn = gr.Button("Reset conversation", size="sm")
                    fill_btn = gr.Button("Fill context window", size="sm")
                    wipe_btn = gr.Button("Wipe long-term memory", size="sm", variant="stop")

        # ---------------- RIGHT: the panels that teach ----------------
        with gr.Column(scale=4, elem_id="side-panel"):
            with gr.Tab("Context"):
                context_md = gr.Markdown(render_context([]))
            with gr.Tab("Memory"):
                memory_md = gr.Markdown(render_memory())
            with gr.Tab("Tools"):
                gr.Markdown(render_tools())
            with gr.Tab("Trace"):
                trace_md = gr.Markdown("### Trace\n_Send a message to see the agent's loop._")
            with gr.Tab("Map"):
                gr.Markdown("Interactive map of the candidate sites, colored by risk. "
                            "Click **Refresh from agent** to run the forecast and plot it.")
                map_plot = gr.Plot(label="Candidate sites")
                refresh_map_btn = gr.Button("Refresh from agent")

    # ---- wiring ----
    outputs = [msg_box, chatbot, messages_state, trace_md, context_md, memory_md, stats_md]
    inputs = [msg_box, chatbot, messages_state, retrieval_mode]

    send_btn.click(on_send, inputs, outputs)
    msg_box.submit(on_send, inputs, outputs)

    reset_btn.click(
        reset_conversation,
        None,
        [chatbot, messages_state, context_md, memory_md, trace_md, stats_md],
    )
    wipe_btn.click(wipe_long_term, None, memory_md)
    fill_btn.click(fill_context, messages_state, [messages_state, context_md])
    refresh_map_btn.click(map_view.build_map, None, map_plot)


if __name__ == "__main__":
    # inbrowser=True pops the tab automatically. share=True would give you a
    # public link (needs internet) -- not needed, everything runs locally.
    demo.launch(theme=gr.themes.Soft(), inbrowser=True)
