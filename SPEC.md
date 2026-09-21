# ASOIAF Wiki Agent — Specification

An agent that answers questions about the A Song of Ice and Fire universe using
**only** content retrieved from [A Wiki of Ice and Fire](https://awoiaf.westeros.org)
(AWOIAF). Runs on a local open-source model via Ollama, hosted on Google Colab
for GPU access.

---

## 1. Goal and constraints

**Goal.** Ask a natural-language question, get an answer grounded in AWOIAF.

**Hard constraints:**

- Answers come from wiki text fetched at query time — never from model memory.
- If the wiki has no relevant page, the agent says so rather than guessing.
- Open-source model only (no hosted API keys).
- Must run on Colab's free tier (T4 GPU, ~16 GB VRAM).

**Explicit non-goals (for v1):**

- No crawling or indexing the whole wiki ahead of time.
- No vector store, no embeddings, no RAG pipeline.
- No persistent deployment — Colab sessions are disposable.

---

## 2. Architecture

Live tool-calling, not retrieval-augmented generation. The model has no wiki
knowledge baked in; it has a *tool* that fetches wiki text on demand.

```
  user question
       │
       ▼
  ┌─────────────┐   decides to call    ┌──────────────┐
  │  ChatOllama │ ───────────────────► │  search_wiki │
  │ llama3.1:8b │ ◄─────────────────── │    (tool)    │
  └─────────────┘   returns page text  └──────┬───────┘
       │                                      │ HTTPS
       ▼                                      ▼
    answer                        awoiaf.westeros.org/api.php
```

The model never reaches the internet. The tool does. Whatever the tool returns
is the entire factual basis for the answer.

**Why this over RAG:** no pre-crawl, no stale index, no embedding costs, and
grounding is structural rather than hoped-for. The trade-off is that questions
spanning many pages ("compare the Targaryen and Baratheon claims") will be
weaker, since the agent only sees what it thought to search for.

---

## 3. Data source

AWOIAF runs MediaWiki, which exposes a machine-readable endpoint:

```
https://awoiaf.westeros.org/api.php
```

No registration, no API key, no rate-limit headers to negotiate. It is the same
content as the human-readable site, returned as JSON.

**Two calls per lookup:**

1. **Find** — `action=query&list=search&srsearch=<query>&format=json`
   Returns candidate page titles ranked by relevance.
2. **Read** — `action=query&prop=extracts&explaintext=1&titles=<title>&format=json`
   Returns that page's text with markup stripped.

**Open question to resolve during implementation:** `prop=extracts` comes from
the TextExtracts extension, which is not installed on every MediaWiki site. If
it returns empty, fall back to `action=parse&prop=wikitext` (raw markup, needs
cleaning) or `prop=revisions&rvprop=content`. Verify this early — it changes how
much parsing work step 2 requires.

**Etiquette.** Send a descriptive `User-Agent` (e.g. `asoiaf-study-bot/0.1`),
set request timeouts, and don't hammer it in a loop. AWOIAF is a fan-run site.

---

## 4. Components to build

### 4.1 `search_wiki` tool

The only tool the agent gets.

| Property | Value |
|---|---|
| Input | `query: str` — a search term, ideally a page title |
| Output | `str` — plain page text, or a clear "no match" message |
| Errors | Never raises; returns a readable message the model can act on |

Requirements:

- Decorated with LangChain's `@tool`.
- The docstring is load-bearing — the model reads it to decide when to call.
  Write it for the model, not for a human reviewer.
- Truncate very long pages (a few thousand characters) before returning.
  Character pages like "Jon Snow" are enormous and will blow the context window
  of an 8B model.
- Return the source page title alongside the text, so answers can cite it.

### 4.2 Agent

- `ChatOllama(model="llama3.1:8b", temperature=0)`.
  Temperature zero: the job is reciting sources, not improvising.
- Built with `create_react_agent` (lives in **`langgraph`**, not `langchain`).
- System prompt must establish:
  - Answer only from `search_wiki` results.
  - Search before answering, every time — even when the answer feels obvious.
  - If the tool returns nothing useful, say the wiki doesn't cover it.
  - Cite the page title used.

### 4.3 Question loop

For v1, a notebook cell taking a question string and printing the answer.
Interactive `input()` loops work in Colab but are awkward; a simple function
call per cell is easier to debug.

---

## 5. Dependencies

`requirements.txt`:

```
langchain
langchain-ollama
requests
pydantic
```

Add `langgraph` if `create_react_agent` isn't already available — depending on
the LangChain version it may or may not come along automatically.

**Deliberately excluded:**

| Package | Why not |
|---|---|
| `wikipedia` | Hardcoded to wikipedia.org; cannot reach AWOIAF |
| `langchain-community` | Only needed for the Wikipedia wrapper we dropped |
| `langchain-openai`, `langchain-anthropic` | Hosted providers; using Ollama |
| `python-dotenv` | No API keys to hide; Colab has a Secrets panel if needed |
| `beautifulsoup4`, `lxml` | Only needed if HTML scraping replaces the API |

Ollama itself is **not** a pip package — it's a server binary, installed
separately (see below).

---

## 6. Colab setup

Set **Runtime → Change runtime type → T4 GPU** first. On CPU this is unusably
slow and there's no reason to use Colab at all.

**Cell 1 — cache models to Drive** (optional, strongly recommended)

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ["OLLAMA_MODELS"] = "/content/drive/MyDrive/ollama_models"
```

Must run *before* the server starts. Costs ~5 GB of Drive, turns a multi-minute
model download into seconds on every future session.

**Cell 2 — install Ollama**

```python
!curl -fsSL https://ollama.com/install.sh | sh
```

**Cell 3 — start the server in the background**

```python
import subprocess, time
subprocess.Popen(["ollama", "serve"],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)
```

`!ollama serve` would block the cell forever — the server never exits. `Popen`
runs it alongside the notebook.

**Cell 4 — pull the model**

```python
!ollama pull llama3.1:8b
```

**Cell 5 — Python packages**

```python
!pip install -q langchain langchain-ollama requests
```

`ChatOllama` then connects to `http://localhost:11434` with no configuration.

---

## 7. Model choice

**`llama3.1:8b`** — supports tool calling, fits comfortably on a T4, well
supported by `langchain-ollama`.

**Tool calling is non-negotiable.** The entire design depends on the model
emitting a structured "call `search_wiki` with X" instruction. Many open models
can't, and will silently hallucinate ASOIAF answers instead — which looks like
it's working while defeating the whole purpose.

Alternatives if llama3.1 disappoints: `qwen2.5:7b`, `mistral-nemo`.

Set expectations honestly: an 8B local model is meaningfully worse at multi-step
tool use than a frontier hosted model. It will sometimes skip the search, or
search with a bad query and give up. The system prompt has to be firmer than
would be necessary with a larger model.

---

## 8. Known risks

| Risk | Mitigation |
|---|---|
| Model answers from memory instead of searching | Firm system prompt; verify by asking about obscure minor characters |
| `prop=extracts` unavailable on AWOIAF | Fall back to `action=parse` or `prop=revisions` |
| Page text overflows context window | Truncate tool output; consider returning only the intro section |
| Colab session dies mid-work | Expected; cache models to Drive, keep code in `.py` files |
| Search finds the wrong page | Return the title in the output so wrong hits are visible |
| Spoilers | Out of scope for v1 — the wiki covers all published books |

---

## 9. Acceptance criteria

v1 is done when:

1. A factual question ("Who is Jon Snow's mother?") returns a correct answer
   drawn from a named wiki page.
2. A question about something absent from the wiki returns a clear "not covered"
   rather than an invention.
3. Tool call traces show `search_wiki` firing on every question.
4. The notebook runs start-to-finish on a fresh Colab runtime without manual
   intervention beyond selecting the GPU.

---

## 10. Possible extensions

- Multi-page questions: let the agent call `search_wiki` several times and
  synthesise.
- Return section headings so the agent can request a specific section.
- Swap the notebook loop for a Gradio chat UI (runs fine inside Colab).
- If single-page lookup proves too limiting, revisit RAG: crawl a subset of the
  wiki, chunk, embed, and retrieve — at the cost of a stale index and real
  setup work.
- Move code from notebook cells into `.py` files under version control, with
  Colab doing `git clone` + `pip install -r requirements.txt`.
