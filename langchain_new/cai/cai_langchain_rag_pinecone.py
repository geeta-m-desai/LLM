import pinecone
from datasets import load_dataset
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone

from langchain_new.cai.utils import get_cai_response, compare_sts_cos_cai_rag_results

"""## Load multiple and process documents"""
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
import tiktoken

# find API key in console at app.pinecone.io
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# find ENV (cloud region) next to API key in console
PINECONE_ENV = os.getenv("PINECONE_ENV")

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)
tokenizer = tiktoken.get_encoding('cl100k_base')


def create_index(embedding):
    index_name = 'langchain-retrieval-augmentation'
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=len(embedding)  # 1536 dim of text-embedding-ada-002
        )
    index = pinecone.GRPCIndex(index_name)
    index.describe_index_stats()
    index.upsert(embedding)


# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def run_llm_rag_cai(texts, user_query):
    llm = OpenAI()
    llm_response = create_index_store(docs, query)
    print(llm_response['result'])
    cai_response = get_cai_response(llm, llm_response['result'])
    compare_sts_cos_cai_rag_results(texts, user_query, llm_response['result'], cai_response, 'pinecone')


def create_index_store(docs, query):
    index_name = 'langchain-retrieval-augmentation'
    embeddings = OpenAIEmbeddings()
    # initialize pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )
    # pinecone.delete_index(index_name)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=tiktoken_len, )
    texts = text_splitter.split_text(docs)
    docs1 = [Document(page_content=t) for t in texts[:4]]
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
    vectorstore = Pinecone.from_documents(docs1, embeddings, index_name=index_name)
    qa_with_sources = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 1, 'score_threshold': 0.3}, return_source_documents=True)
    )
    return qa_with_sources(query)


if __name__ == "__main__":
    dataset = load_dataset("CarperAI/openai_summarize_tldr")
    train_dataset = dataset.data['train']
    test_dataset = dataset.data['test']
    valid_dataset = dataset.data['valid']
    docs = train_dataset[0][1]
    docs = docs.as_py()
    query = "Why was I disappointed When I weighed myself as normal on Monday?"
    run_llm_rag_cai(docs, query)
