import pinecone
from datasets import load_dataset
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone

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


def run_llm_rag_cai(text, user_query):
    # splitting the text into
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=tiktoken_len, )
    texts = text_splitter.split_text(text)
    docs1 = [Document(page_content=t) for t in texts[:4]]
    embedding = OpenAIEmbeddings()
    res = embedding.embed_documents(texts)
    len(res), len(res[0])
    create_index(res)


def create_vector_store(data):
    from uuid import uuid4
    index_name = 'langchain-retrieval-augmentation'
    embedding = OpenAIEmbeddings()
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536  # 1536 dim of text-embedding-ada-002
        )
    index = pinecone.GRPCIndex(index_name)
    index.describe_index_stats()
    batch_limit = 100

    texts = []
    metadatas = []

    for i, record in enumerate(data):
        # first get metadata fields for this record

        metadata = {
            'id': "text_id"

        }
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,
                                                       length_function=tiktoken_len, )
        texts = text_splitter.split_text(data)
        # create individual metadata dicts for each chunk
        docs1 = [Document(page_content=t) for t in texts[:4]]
        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(texts)]
        metadatas.extend(record_metadatas)
        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embedding.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []


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
        retriever=vectorstore.as_retriever()
    )
    print(qa_with_sources(query))


if __name__ == "__main__":
    dataset = load_dataset("CarperAI/openai_summarize_tldr")
    train_dataset = dataset.data['train']
    test_dataset = dataset.data['test']
    valid_dataset = dataset.data['valid']
    docs = train_dataset[0][1]
    docs = docs.as_py()
    query = "What did 30 years old do?"
    create_index_store(docs, query)
    # run_llm_rag_cai(docs, query)
