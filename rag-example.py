import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. Preparation: Create a dummy local document ---
# In real usage, you can just point to an existing text file.
with open("company_policy.txt", "w") as f:
    f.write("The company remote work policy allows employees to work from anywhere up to 3 days a week. "
            "Core collaboration hours are between 10:00 AM and 3:00 PM EST. "
            "Employees receive a $500 home-office stipend upon onboarding.")

# --- 2. Load and Chunk the Document ---
loader = TextLoader("company_policy.txt")
docs = loader.load()

# Split large text into smaller chunks so the LLM can read them effectively
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# --- 3. Set Up Local Vector DB and Embeddings ---
# We use Ollama to generate numerical vector representations (embeddings) of our text
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Chroma is a local, in-memory vector database
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 1}) # Fetch the single most relevant chunk

# --- 4. Initialize Local Model and Prompt ---
model = ChatOllama(model="llama3", temperature=0)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question using only the provided context. If you don't know, say you don't know.\n\nContext:\n{context}"),
    ("user", "{question}")
])

# --- 5. RAG Pipeline Logic ---
# Retrieve documents based on the question
question = "What is the policy regarding remote work and stipends?"
relevant_docs = retriever.invoke(question)
context_text = "\n".join([doc.page_content for doc in relevant_docs])

# Construct the chain
chain = prompt_template | model | StrOutputParser()

# --- 6. Stream the Output ---
# Instead of chain.invoke(), we use chain.stream() to print tokens as they generate
print("Streaming Answer: ", end="", flush=True)
for chunk in chain.stream({"context": context_text, "question": question}):
    print(chunk, end="", flush=True)
print()

# Clean up the dummy file
if os.path.exists("company_policy.txt"):
    os.remove("company_policy.txt")

