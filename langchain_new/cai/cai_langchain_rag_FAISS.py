import pandas as pd
from datasets import load_dataset
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from langchain_new.cai.utils import get_cai_response, compare_sts_cos_cai_results, write_file, \
    compare_sts_cos_cai_rag_results, chain_of_verification, query_llm, create_questions_llm

"""## Load multiple and process documents"""
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def run_llm_rag_cai(text, user_query):
    # splitting the text into
    llm = OpenAI()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    docs1 = [Document(page_content=t) for t in texts[:4]]
    embedding = OpenAIEmbeddings()
    db = FAISS.from_documents(docs1, embedding)
    results1 = db.similarity_search_with_score(user_query, k=1, score_threshold=0.6)
    llm_response = query_llm(llm, user_query, results1)
    print("llm_response ", llm_response, "\n")
    if len(results1) == 0:
        cai_response = (get_cai_response(llm, user_query))
        compare_sts_cos_cai_rag_results(texts, user_query, "This is generic answer as relevant information could not be obtained. "+llm_response['text'], cai_response, 'faiss')

    else:
        final_res = chain_of_verification(llm, user_query, results1, llm_response['text'])
        print("\n final_res", final_res)
        cai_response = get_cai_response(llm, final_res['text'])
        compare_sts_cos_cai_rag_results(texts, user_query, final_res['text'], cai_response, 'faiss')


if __name__ == "__main__":
    quest_lst = pd.read_csv("questions.csv")
    for i in range(51, 1000):
        run_llm_rag_cai(quest_lst['orig_text'][i], quest_lst['quest_text'][i])
