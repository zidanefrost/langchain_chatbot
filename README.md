# langchain_chatbot

> **UCL side-project** exploring LangChain, OpenAI embeddings, and Chroma-backed vector search.

### Goal
Turn a plain-text corpus into a mini-chatbot that can answer questions with context-aware retrieval.

### Quick start
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=<your key>
python chatbot.py
