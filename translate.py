from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# 1. Initialize the local Ollama model
model = ChatOllama(model="llama3", temperature=0)

# 2. Define the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates text from English to {language}. Reply with only the translation."),
    ("user", "{text}"),
])

# 3. Chain the components together using the pipe operator (|)
chain = prompt_template | model | StrOutputParser()

if __name__ == "__main__":
    # 4. Run the chain locally
    print(chain.invoke({"language": "Italian", "text": "Hello, how are you today?"}))
    # Example output: Ciao, come stai oggi?
