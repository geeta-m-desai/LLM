
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain_experimental import AutoGPT
from langchain.chat_models import ChatOpenAI
import os
from langchain_new.tools import tools

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
os.environ["OPENAI_API_KEY"] = ""
# Initialize the vectorstore as empty
import faiss

embedding_size = 1536 #openai embeddings has 1536 dimensions
index = faiss.IndexFlatL2(embedding_size) #Index that stores the full vectors and performs exhaustive search
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    memory=vectorstore.as_retriever()
)
# Set verbose to be true
agent.chain.verbose = True
agent.run(["write a weather report for Helsinki today"])
