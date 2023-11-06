import tiktoken
from datasets import load_dataset
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from llama_index import ServiceContext, OpenAIEmbedding, PromptHelper
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import TokenTextSplitter

from langchain_new.cai.utils import get_cai_response, compare_sts_cos_cai_rag_results

"""## Load multiple and process documents"""
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def run_llm_rag_cai(text, user_query):
    # splitting the text into
    llm = OpenAI()
    embed_model = OpenAIEmbedding()
    text_splitter = TokenTextSplitter(
        separator=" ",
        chunk_size=1024,
        chunk_overlap=20,
        backup_separators=["\n"],
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    )
    node_parser = SimpleNodeParser.from_defaults(
        text_splitter=text_splitter
    )
    prompt_helper = PromptHelper(
        context_window=4096,
        num_output=256,
        chunk_overlap_ratio=0.1,
        chunk_size_limit=None
    )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
        prompt_helper=prompt_helper
    )

    documents = SimpleDirectoryReader(input_dir='data').load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context
    )
    index.storage_context.persist()
    response = index.as_query_engine().query(query)
    # retriever = index.as_retriever()
    # llm = OpenAI()
    # qa_chain = (RetrievalQA.from_chain_type(llm=llm,
    #                                         chain_type="stuff",
    #                                         retriever=retriever,
    #                                         return_source_documents=True))
    # llm_response = (qa_chain(user_query))
    llm_response = {'results': str(response)}
    print(llm_response)

    cai_response = get_cai_response(llm, llm_response['results'])
    print(cai_response)
    # cai_response = get_cai_response(llm, llm_response['results'])
    compare_sts_cos_cai_rag_results(text, user_query, llm_response['results'], cai_response,"llamaindex")


if __name__ == "__main__":
    dataset = load_dataset("CarperAI/openai_summarize_tldr")
    train_dataset = dataset.data['train']
    test_dataset = dataset.data['test']
    valid_dataset = dataset.data['valid']
    docs = train_dataset[0][1]
    docs = docs.as_py()

    query = "What did 30 years old do?"
    run_llm_rag_cai(docs, query)
