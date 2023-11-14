from datasets import load_dataset
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from langchain_new.cai.utils import get_cai_response, compare_sts_cos_cai_results, write_file, \
    compare_sts_cos_cai_rag_results, chain_of_verification

"""## Load multiple and process documents"""
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def run_llm_rag_cai(text, user_query):
    # splitting the text into
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    docs1 = [Document(page_content=t) for t in texts[:4]]
    embedding = OpenAIEmbeddings()
    db = FAISS.from_documents(docs1, embedding)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 1, 'score_threshold': 0.3})
    llm = OpenAI()
    qa_chain = (RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            retriever=retriever,
                                            return_source_documents=True))
    llm_response = (qa_chain(user_query))
    print(llm_response,"\n")
    source_documents = llm_response['source_documents']
    if str(source_documents[0].page_content).upper() == 'TL;DR:':
        cai_response = (get_cai_response(llm, user_query))
        compare_sts_cos_cai_rag_results(texts, query, llm_response['result'], cai_response, 'faiss')
    else:
        final_res = chain_of_verification(llm, user_query, source_documents, llm_response['result'])
        print("\n final_res", final_res)
        cai_response = get_cai_response(llm, final_res['text'])
        compare_sts_cos_cai_rag_results(texts, query, final_res['text'], cai_response, 'faiss')


if __name__ == "__main__":
    dataset = load_dataset("CarperAI/openai_summarize_tldr")
    train_dataset = dataset.data['train']
    test_dataset = dataset.data['test']
    valid_dataset = dataset.data['valid']
    docs = train_dataset[0][1]
    docs = docs.as_py()
    query = "How to show disresepct to others?"
    run_llm_rag_cai(docs, query)
