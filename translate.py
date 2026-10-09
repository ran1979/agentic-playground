from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# 1. Initialize the Local Ollama Model
model = ChatOllama(model="llama3", temperature=0)

# 2. Define the Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates text from English to {language}."),
    ("user", "{text}")
])

# 3. Create the Output Parser
output_parser = StrOutputParser()

# 4. Chain the components together using the pipe operator (|)
chain = prompt_template | model | output_parser

# 5. Run the chain locally
response = chain.invoke({"language": "Italian", "text": "Hello, how are you today?"})

print(response)
# Output: Ciao, come stai oggi?

