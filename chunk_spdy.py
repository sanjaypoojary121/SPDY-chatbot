from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Load the website text
with open("mite_website_full_content.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# Step 2: Setup chunking logic
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # characters per chunk (adjustable)
    chunk_overlap=50,      # overlap between chunks to preserve context
    separators=["\n\n", "\n", ".", " ", ""],  # split by logical breaks
)

# Step 3: Create chunks
chunks = text_splitter.split_text(full_text)

# Step 4: Save chunks to a file
with open("mite_website_chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"\n--- Chunk {i+1} ---\n{chunk}\n")

print(f"✅ Total chunks created: {len(chunks)}")
