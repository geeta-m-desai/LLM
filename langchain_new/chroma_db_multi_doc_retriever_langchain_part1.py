
import os

os.environ["OPENAI_API_KEY"] = ""

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

"""## Load multiple and process documents"""

# Load and process the text files
# loader = TextLoader('single_text_file.txt')
loader = DirectoryLoader('./new_articles/', glob="./*.txt", loader_cls=TextLoader)

documents = loader.load()

#splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

len(texts)

texts[3]

"""## create the DB"""

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

## here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

# persiste the db to disk
vectordb.persist()
vectordb = None

# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

"""## Make a retriever"""

retriever = vectordb.as_retriever()

docs = retriever.get_relevant_documents("How much money did Pando raise?")

len(docs)

retriever = vectordb.as_retriever(search_kwargs={"k": 2})

retriever.search_type

retriever.search_kwargs

"""## Make a chain"""

# create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)

## Cite sources
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

# full example
query = "How much money did Pando raise?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

# break it down
query = "What is the news about Pando?"
llm_response = qa_chain(query)
# process_llm_response(llm_response)
llm_response

query = "Who led the round in Pando?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "What did databricks acquire?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "What is generative ai?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "Who is CMA?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

qa_chain.retriever.search_type , qa_chain.retriever.vectorstore

print(qa_chain.combine_documents_chain.llm_chain.prompt.template)


# To cleanup, you can delete the collection
vectordb.delete_collection()
vectordb.persist()


import os

os.environ["OPENAI_API_KEY"] = ""

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

persist_directory = 'db'
embedding = OpenAIEmbeddings()

vectordb2 = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding,
                   )

retriever = vectordb2.as_retriever(search_kwargs={"k": 2})

# Set up the turbo LLM
turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo'
)

# create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)

## Cite sources
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

# full example
query = "How much money did Pando raise?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

"""### Chat prompts"""

print(qa_chain.combine_documents_chain.llm_chain.prompt.messages[0].prompt.template)

print(qa_chain.combine_documents_chain.llm_chain.prompt.messages[1].prompt.template)

