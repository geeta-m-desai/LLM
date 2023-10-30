from datasets import load_dataset
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain_new.cai.utils import get_cai_response, compare_sts_cos_cai_results

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
    vectordb = Chroma.from_documents(documents=docs1,
                                     embedding=embedding,
                                     )

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    llm = OpenAI()
    qa_chain = (RetrievalQA.from_chain_type(llm=llm,
                 chain_type="stuff",
                 retriever=retriever,
                 return_source_documents=True))
    llm_response = (qa_chain(user_query))
    print(llm_response)
    cai_response = get_cai_response(llm, llm_response['result'])
    final_cai_outcome = compare_sts_cos_cai_results(llm_response['result'], cai_response)
    print(final_cai_outcome)


if __name__ == "__main__":
    dataset = load_dataset("CarperAI/openai_summarize_tldr")
    train_dataset = dataset.data['train']
    test_dataset = dataset.data['test']
    valid_dataset = dataset.data['valid']
    docs = train_dataset[0][1]
    docs = docs.as_py()
    query = "What did 30 years old do?"
    run_llm_rag_cai(docs, query)
