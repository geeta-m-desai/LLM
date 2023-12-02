import pandas as pd
import pinecone
from datasets import load_dataset
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone

from langchain_new.cai.utils import get_cai_response, compare_sts_cos_cai_rag_results, query_llm, chain_of_verification, \
    create_questions_llm

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
    create_index_store(llm, texts, user_query)


def create_index_store(llm, docs, query):
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
    results1 = vectorstore.similarity_search_with_score(query, k=1)
    llm_response = query_llm(llm, query, results1)
    print("llm_response ", llm_response, "\n")
    if len(results1) == 0 or results1[0][1] > 0.6:
        cai_response = (get_cai_response(llm, query))
        compare_sts_cos_cai_rag_results(texts, query,
                                        "This is generic answer as relevant information could not be obtained. " +
                                        llm_response['text'], cai_response, 'pinecone')

    else:
        final_res = chain_of_verification(llm, query, results1, llm_response['text'])
        print("\n final_res", final_res)
        cai_response = get_cai_response(llm, final_res['text'])
        compare_sts_cos_cai_rag_results(texts, query, final_res['text'], cai_response, 'pinecone')


if __name__ == "__main__":
    quest_lst = pd.read_csv("questions.csv")
    print("Question is ", quest_lst['quest_text'][38], "\n")
    run_llm_rag_cai(quest_lst['orig_text'][38], "How to avoid people who drink?")
    # for i in range(1, 50):
    # run_llm_rag_cai(quest_lst['orig_text'][i], quest_lst['quest_text'][i])
