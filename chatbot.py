from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os, sys

# load docs
with open("docs/my_doc.txt") as f:
    raw = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_text(raw)

# embed + store
emb = OpenAIEmbeddings()
vectordb = Chroma.from_texts(docs, embedding=emb, persist_directory="db")
retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.0),
                                 chain_type="stuff",
                                 retriever=retriever)

print("Vectorstore ready ✅")
