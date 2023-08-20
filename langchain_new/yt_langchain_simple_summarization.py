import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = ""

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.indexes import VectorstoreIndexCreator

llm = OpenAI(temperature=0)

text_splitter = CharacterTextSplitter()
loader = PyPDFLoader("1808.08745.pdf")
pages = loader.load_and_split()
index = VectorstoreIndexCreator().from_loaders([loader])
#just for creating the vector store. It can't actually be used as a retriever.
#VectorstoreIndexCreator(vectorstore_cls=Chroma, embedding=embeddings, vectorstore_kwargs={ "persist_directory": "/persistance/directory"}).from_loaders([loader])
print("index summary", index.query("Summarize the general content of this document.",
                                   retriever_kwargs={"search_kwargs": {"filter": {"source": "1808.08745.pdf"}}},
                                   top_k=10))

# Create embeddings for the pages and insert into Chroma database
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embeddings, top_k=10)

# Initialize the OpenAI module, load and run the summarize chain
llm = OpenAI(temperature=0)
chain = load_summarize_chain(llm, chain_type="stuff")
search = vectordb.similarity_search(" ")
summary = chain.run(input_documents=search, question="Write a summary within 150 words.")
print("vector summary", summary)

# output_summary = chain.run(pages)
# print("output summary", output_summary)
#
# evil_qa_prompt = PromptTemplate(
#     template="""You are legal advisor. Rephrase input for legal policies.
#
# Question: {question}
#
# Evil answer:""",
#     input_variables=["question"],
# )
#
#
#
# evil_qa_chain = LLMChain(llm=llm, prompt=evil_qa_prompt)
# from langchain.chains.constitutional_ai.base import ConstitutionalChain
# from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
#
# ethical_principle = ConstitutionalPrinciple(
#     name="Ethical Principle",
#     critique_request="The model should only talk about  ethical and legal things.",
#     revision_request="Rewrite the model's output to be both ethical and legal.",
# )
# postprocess_styling="Ada Lovelace"
# styling_principal = ConstitutionalPrinciple(
#                 name=f'{postprocess_styling} Principle',
#                 critique_request=f'Identify specific ways in which the model\'s response is not in the style of {postprocess_styling}.',
#                 revision_request=f'Please rewrite the model response to be in the style of {postprocess_styling}.',
#             )
# constitutional_principles = [ethical_principle, styling_principal]
# constitutional_chain = ConstitutionalChain.from_llm(
#     chain=evil_qa_chain,
#     constitutional_principles=[ethical_principle],
#     llm=llm,
#     verbose=True,
# )
#
# constitutional_chain = ConstitutionalChain.from_llm(
#     chain=evil_qa_chain,
#     constitutional_principles=constitutional_principles,
#     llm=llm,
#     verbose=True,
# )
#
# constitutional_chain.run(question=summary)
