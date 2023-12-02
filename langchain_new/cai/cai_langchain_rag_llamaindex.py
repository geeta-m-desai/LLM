import pandas as pd
import tiktoken
from datasets import load_dataset
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from llama_index import ServiceContext, OpenAIEmbedding, PromptHelper, StorageContext, load_index_from_storage
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import TokenTextSplitter

from langchain_new.cai.utils import get_cai_response, compare_sts_cos_cai_rag_results, chain_of_verification, query_llm
from pathlib import Path
from llama_index import download_loader

from llama_index.indices.vector_store.retrievers.retriever import VectorIndexRetriever

from llama_index.response_synthesizers.factory import get_response_synthesizer

from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine

from llama_index.indices.postprocessor.node import SimilarityPostprocessor

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
        prompt_helper=prompt_helper,

    )

    # documents = SimpleDirectoryReader(input_dir='data').load_data()
    PandasCSVReader = download_loader("PandasCSVReader")
    #
    loader = PandasCSVReader()
    documents = loader.load_data(file=Path('questions-llama.csv'))
    #storage_context = StorageContext.from_defaults(persist_dir='storage')
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        #storage_context=storage_context,

    )
    # index.set_index_id("rag-cai-index")
    # rebuild storage context

    # load index
    # index = load_index_from_storage(storage_context,index_id="rag-cai-index")
    # if index:
    #     print("Index found")
    # index.storage_context.persist()
    # if VectorStoreIndex.index_id != "rag-cai-index":
    #     index = VectorStoreIndex.from_documents(
    #         documents,
    #         service_context=service_context
    #     )
    #     index.storage_context.persist()
    #     index.set_index_id("rag-cai-index")
    # else:
    #     print("index already created")
    #     index = VectorStoreIndex.index_id

    # response = index.as_query_engine(similarity_top_k=3).query(user_query)
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=1,
    )

    # Configure response synthesizer
    response_synthesizer = get_response_synthesizer()

    # Assemble query engine with postprocessors
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.5)
        ]
    )

    # Execute the query
    response = query_engine.query(user_query)
    # retriever = index.as_retriever()
    # llm = OpenAI()
    # qa_chain = (RetrievalQA.from_chain_type(llm=llm,
    #                                         chain_type="stuff",
    #                                         retriever=retriever,
    #                                         return_source_documents=True))
    # llm_response = (qa_chain(user_query))
    llm_response = {'results': str(response)}
    # print("QA Response ... ", llm_response)

    # Get sources
    # print(response.source_nodes)
    print("Formatted Response \n ", response.get_formatted_sources())
    if str(response) == 'Empty Response':
        cai_response = (get_cai_response(llm, user_query))
        compare_sts_cos_cai_rag_results(text, user_query,
                                        "This is generic answer as relevant information could not be obtained. " +
                                        llm_response['results'], cai_response, 'llamaindex')

    else:
        cai_response = get_cai_response(llm, llm_response['results'])
        print(cai_response)
        # cai_response = get_cai_response(llm, llm_response['results'])
        compare_sts_cos_cai_rag_results(text, user_query, llm_response['results'], cai_response, "llamaindex")


if __name__ == "__main__":
    quest_lst = pd.read_csv("questions.csv")
    # print("quest", quest_lst['orig_text'][24], "\n", quest_lst['quest_text'][24])
    #
    run_llm_rag_cai(quest_lst['orig_text'][38], quest_lst['quest_text'][38])

    # for i in range(100, 200):
    #     run_llm_rag_cai(quest_lst['orig_text'][i], quest_lst['quest_text'][i])
