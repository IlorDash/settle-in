# SettleIn

Telegram bot that helps immigrants in Serbia navigate administrative procedures, understand documents, and overcome language barriers.

Built with Python, LangChain, LangGraph, and ChromaDB.

## What It Does

- **Knowledge Q&A** — ask questions about Serbian residency, work permits, banking, health insurance, tax registration, and daily procedures. Answers are sourced from a curated knowledge base using RAG (Retrieval-Augmented Generation)
- **Translation** — translate between Serbian, English, and Russian, including request phrasings like "Как будет спасибо по-сербски"
- **Document reading** — photograph a Serbian bill, letter, or form and the bot says what it is, pulls out the dates, amounts, and reference numbers, and explains what to do next
- **Smart routing** — a locally trained classifier picks the right agent in milliseconds, falling back to an LLM only when it is unsure, and rejecting questions outside the bot's scope. A photo skips classification entirely: the modality already names the agent
- **Conversational memory** — each chat keeps its own history, so follow-ups like "and in Latin?" are understood, and it is stored on disk so a restart does not wipe it. Every agent reads it, so a question about a document the bot has already explained is answered from that explanation. `/reset` forgets the conversation without touching saved preferences
- **Standing preferences** — `/pref add Write Serbian translations in Cyrillic` saves a rule the bot applies to later answers
- **Chat export** — `/export` sends the recent messages back as a text file, which is how a bug gets reported with the bot's exact wording instead of a retyped approximation
- **Feedback buttons** — 👍/👎 under each answer, recorded with the predicted intent as labelled data for retraining the classifier
- **Input protection** — message validation, per-user rate limiting, and graceful error handling for LLM timeouts and API failures, with a catch-all handler so even an unexpected crash gets a reply rather than silence

## Architecture

```
User (Telegram app)
    │
    │ HTTPS (Telegram Bot API)
    ▼
Bot Handler Layer (handlers.py)
    │ validates input
    │ checks rate limit
    ├─────── photo ────────────────────────────┐
    │ text                                     ▼
    ▼                                   Multimodal Agent
Orchestrator (orchestrator.py)          (multimodal_agent.py)
    │ classifies user intent                   │
    │ routes to correct agent                  │
    ├──────────────────────┐                   │
    ▼                      ▼                   │
RAG Agent              Translation Agent       │
(rag_agent.py)         (translation_agent.py)  │
    │                      │                   │
    ▼                      │                   │
ChromaDB                   │                   │
(vector store)             │                   │
    │                      │                   │
    └──────────┬───────────┴───────────────────┘
               ▼
        Response back to user via Telegram
```

### Orchestrator Graph

The orchestrator is built as a LangGraph StateGraph with intent-based routing:

![Orchestrator Graph](assets/orchestrator_graph.png)

The multimodal agent is deliberately **not** a node in this graph. Routing to it
happens one level earlier, in the Telegram handler layer, because an image
carries no text for `classify_intent` to work on. Keeping it outside also keeps
photos out of the graph's state, where a base64 image would be written into
every saved checkpoint. Regenerate the diagram with
`python -m scripts.visualize_graph`.

## Project Structure

```
settle-in/
├── src/
│   ├── bot/
│   │   ├── app.py              # Bot entry point, webhook/polling startup
│   │   ├── handlers.py         # /start, /help, /pref, /reset, /export, messages, photos
│   │   ├── feedback.py         # Appends thumbs up/down to a JSONL store
│   │   ├── transcript.py       # Renders stored messages as readable text
│   │   └── middleware.py       # Input validation, image checks, rate limiting
│   ├── agents/
│   │   ├── orchestrator.py     # LangGraph intent router, memory, preferences
│   │   ├── intent_classifier.py# Loads the trained classifier artifact
│   │   ├── rag_agent.py        # Knowledge retrieval chain
│   │   ├── multimodal_agent.py # Reads a photographed document
│   │   └── translation_agent.py
│   ├── knowledge/
│   │   ├── loader.py           # Document loading and text chunking
│   │   └── vectorstore.py      # ChromaDB setup and retriever
│   └── config.py               # Environment variable management
├── scripts/
│   ├── chat.py                 # Local CLI harness, no Telegram needed
│   ├── probe_vision.py         # Measures what the vision model can read
│   └── train_intent_classifier.py
├── data/
│   ├── knowledge_base/         # 8 text documents on Serbian procedures
│   ├── feedback.jsonl          # Collected thumbs up/down (created on first vote)
│   └── checkpoints.sqlite      # Chat history and /pref rules (created on first use)
├── tests/
│   ├── conftest.py             # Shared fixtures (mock Telegram, mock orchestrator)
│   ├── unit/                   # 113 unit tests
│   ├── integration/            # 27 integration tests
│   └── evals/                  # 10 opt-in tests that call the real OpenAI API
├── Dockerfile
├── pyproject.toml
└── .env.example
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Bot framework | python-telegram-bot 21+ |
| AI orchestration | LangChain + LangGraph |
| LLM | OpenAI GPT-4o-mini (GPT-4o for translation and document reading) |
| Intent classifier | scikit-learn TF-IDF + MLP, trained locally |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | ChromaDB |
| Chat memory | LangGraph checkpointer backed by SQLite (a file, not a server) |
| Testing | pytest, pytest-asyncio, pytest-cov |
| Linting | ruff |

## Quick Start

```bash
# Clone and enter the project
git clone https://github.com/yourusername/settle-in.git
cd settle-in

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env with your Telegram bot token and OpenAI API key

# Run the bot (polling mode for local development)
python -m src.bot.app
```

On first startup, the bot automatically builds the ChromaDB vector store from the knowledge base documents. This calls the OpenAI Embeddings API and takes about 10-20 seconds.

## Local Development

### Talking to the bot without Telegram

`scripts/chat.py` runs the whole orchestrator in the terminal. It prints the predicted intent and the classifier's confidence next to each reply, which makes it the fastest way to check routing:

```bash
python -m scripts.chat
```

It also mirrors `/pref`, so preferences can be tested locally. What it *cannot* test is anything Telegram-specific: inline buttons, feedback taps, or message formatting all need a real bot.

### Reading back what the bot remembers

`/export` in the chat sends the recent messages as a text file, and `/export 50` changes how many. Pair it with `/reset` when retesting a failure. Memory is per chat, so a correction stays in context and the next attempt answers from that correction rather than from the document, which makes the retry look like a fix.

### Running a second bot for testing

Test changes against Telegram without touching the deployed bot by running a separate bot locally. Telegram allows only one process per token, so the development bot needs its own.

1. Message [@BotFather](https://t.me/BotFather), send `/newbot`, and name it something like `SettleInDev`. Copy the token it gives you.
2. Start the bot with that token set in your shell:

```powershell
# Windows PowerShell
$env:TELEGRAM_BOT_TOKEN="<dev-token-from-botfather>"
python -m src.bot.app
```

```bash
# Linux/Mac
TELEGRAM_BOT_TOKEN="<dev-token-from-botfather>" python -m src.bot.app
```

A variable set in the shell takes priority over `.env`, so the production token stays where it is and there is no file to edit or remember to change back. `BOT_MODE` defaults to `polling`, so no public URL is needed — the bot asks Telegram for updates from your machine.

Open the new bot in Telegram and it behaves exactly like production, reading the code in your working tree.

### Running with Docker

```bash
docker build -t settlein-bot .
docker run --env-file .env settlein-bot
```

## Deployment

The bot is deployed on [Railway](https://railway.app) using Docker with webhook mode.

### Environment Variables

| Variable | Required | Default | Description |
| -------- | -------- | ------- | ----------- |
| `TELEGRAM_BOT_TOKEN` | Yes | — | Bot token from BotFather |
| `OPENAI_API_KEY` | Yes | — | OpenAI API key for LLM and embeddings |
| `BOT_MODE` | No | `polling` | `polling` for local dev, `webhook` for production |
| `WEBHOOK_URL` | No | `""` | Public HTTPS URL for webhook mode |
| `PORT` | No | `8443` | Port for the webhook server |
| `CHROMA_PERSIST_DIR` | No | `./data/chroma_db` | Path where ChromaDB stores vector data |
| `FEEDBACK_PATH` | No | `./data/feedback.jsonl` | File collecting thumbs up/down verdicts |
| `CHECKPOINT_PATH` | No | `./data/checkpoints.sqlite` | File holding chat history and `/pref` rules |

### Persistent storage (required on Railway)

A container's filesystem is recreated on every deploy. The bot writes three
things that must outlive it: the vector store, the collected feedback, and each
chat's history and preferences. Without a volume, every deploy erases all three
and the knowledge base is re-embedded from scratch, which costs money.

Attach a Railway **Volume**, mount it at `/data`, and point the three path
variables into it:

```
CHROMA_PERSIST_DIR=/data/chroma_db
FEEDBACK_PATH=/data/feedback.jsonl
CHECKPOINT_PATH=/data/checkpoints.sqlite
```

The last of these is a SQLite file. SQLite is an embedded database: a single
file plus a library that already ships with Python, with no server to run and
no port to expose. That is what keeps this within the project's "no PostgreSQL,
no Redis" constraint while still surviving a restart.

Setting these variables is not optional housekeeping. The code reads whatever
path it is given, so leaving the defaults means the bot writes inside the
container and the data disappears at the next deploy exactly as before.

### Polling vs Webhook

- **Polling** (local development): the bot asks Telegram "any new messages?" every few seconds. No public URL needed.
- **Webhook** (production): Telegram pushes messages to the bot's HTTPS endpoint. Faster and more efficient. Requires a public URL with TLS — Railway provides this automatically.

## Testing

140 tests (113 unit + 27 integration) at 90% code coverage, plus 10 opt-in evals.

```bash
# Run all tests (free and offline — evals are excluded)
pytest tests/ -v

# Run with coverage report
pytest --cov=src tests/

# Run only unit tests
pytest tests/unit/ -v

# Run the evals — these call the real OpenAI API and cost money
pytest -m eval tests/evals -v
```

**Unit tests** verify each function in isolation by mocking external dependencies (OpenAI API, Telegram API, ChromaDB). **Integration tests** verify that components work together — for example, that a user message flows correctly through the orchestrator to the right agent.

**Evals** are a third tier that exists because the first two mock the language model, and so are blind to what it actually replies. They send real requests to check things no mocked test can see: that an explicit "in Latin" request overrides a saved Cyrillic preference, that the agent never hands back the untranslated source word, and that a bare follow-up is resolved from conversation history. Because they cost money, a pytest marker excludes them from every ordinary run and `-m eval` opts back in.
