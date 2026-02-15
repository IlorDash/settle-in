# SettleIn

Telegram bot that helps immigrants in Serbia navigate administrative procedures, understand documents, and overcome language barriers.

Built with Python, LangChain, and ChromaDB as a Master's Thesis project (Singidunum University).

## What It Does

- **Ask questions** about Serbian residency, work permits, banking, and daily procedures — get answers sourced from a curated knowledge base
- **Translate** between Serbian and English
- **Smart routing** — an orchestrator detects your intent and picks the right AI agent automatically

## Architecture

```
Telegram Client ──► Bot Handlers ──► Orchestrator ──► RAG Agent ──► ChromaDB
                                                  └──► Translation Agent
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Bot framework | python-telegram-bot 21+ |
| AI orchestration | LangChain |
| Vector store | ChromaDB |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Testing | pytest + pytest-asyncio |

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

# Run the bot
python -m src.bot.app
```

## Testing

```bash
pytest tests/ -v
```
