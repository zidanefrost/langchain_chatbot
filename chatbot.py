from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. read local file
with open("docs/my_doc.txt") as f:
    text = f.read()

# 2. split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(text)

print(f"Loaded {len(chunks)} chunks")
