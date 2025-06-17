# LangChain Retrieval-Augmented Chatbot

🚀 A small self-built QA chatbot using **LangChain**, **OpenAI**, and **ChromaDB (FAISS)** — built as part of my personal deep dive into RAG pipelines and LLM-augmented search.

---

## 💡 What it Does

- ✅ Loads a domain-specific document (`my_doc.txt`)
- ✅ Splits into overlapping chunks using `RecursiveCharacterTextSplitter`
- ✅ Embeds with `OpenAIEmbeddings`
- ✅ Stores vectors in **Chroma (FAISS backend)**
- ✅ Runs **retrieval-based Q&A** via `RetrievalQA` and `OpenAI()`
- ✅ Interacts via a basic terminal CLI

---

## 🧰 Tech Stack

| Tool        | Use                    |
|-------------|-------------------------|
| `LangChain` | Q&A chain, vector interface |
| `OpenAI API` | Embeddings + GPT inference |
| `ChromaDB`  | Local vector database (FAISS) |
| `Python`    | Everything |

---

## ⚙️ Setup & Run

```bash
git clone https://github.com/yourhandle/langchain_chatbot_ucl.git
cd langchain_chatbot_ucl
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your-api-key-here

python chatbot.py
