import tempfile
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

POLICY_TEXT = (
    "The company remote work policy allows employees to work from anywhere up to 3 days a week. "
    "Core collaboration hours are between 10:00 AM and 3:00 PM EST. "
    "Employees receive a $500 home-office stipend upon onboarding."
)


def build_retriever(path: Path):
    """Load a text file, chunk it, embed it, and return a retriever over the chunks."""
    docs = TextLoader(str(path)).load()

    # Split large text into smaller chunks so the LLM can read them effectively
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Ollama generates the embeddings; Chroma is a local in-memory vector DB
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 1})  # single most relevant chunk


def format_docs(docs) -> str:
    return "\n".join(doc.page_content for doc in docs)


def main():
    # In real usage, point at an existing file instead of this temporary one.
    with tempfile.TemporaryDirectory() as tmp:
        policy_file = Path(tmp) / "company_policy.txt"
        policy_file.write_text(POLICY_TEXT)
        retriever = build_retriever(policy_file)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question using only the provided context. "
                   "If you don't know, say you don't know.\n\nContext:\n{context}"),
        ("user", "{question}"),
    ])
    model = ChatOllama(model="llama3", temperature=0)

    # LCEL RAG chain: retrieve -> format -> prompt -> model -> string
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    question = "What is the policy regarding remote work and stipends?"
    print("Streaming Answer: ", end="", flush=True)
    for token in chain.stream(question):  # stream() prints tokens as they generate
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
