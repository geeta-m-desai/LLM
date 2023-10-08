

import os

os.environ["OPENAI_API_KEY"] = ""


"""## Parent Document Retriever

2 ways to use it:

1. Return full docs from smaller chunks look up
2. Return bigger chunks for smaller chunks look up
"""

from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever

## Text Splitting & Docloader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.document_loaders import TextLoader

# from langchain.embeddings.openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings()

"""## BGE Embeddings"""

from langchain.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-small-en-v1.5"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'},
    encode_kwargs=encode_kwargs
)

"""## Data prep

"""

loaders = [
    TextLoader('/content/blog_posts/blog.langchain.dev_announcing-langsmith_.txt'),
    TextLoader('/content/blog_posts/blog.langchain.dev_benchmarking-question-answering-over-csv-data_.txt'),
]
docs = []
for l in loaders:
    docs.extend(l.load())

len(docs)

docs[0]

"""## 1. Retrieving full documents rather than chunks
In this mode, we want to retrieve the full documents.

This is good to use if you initial full docs aren't too big themselves and you aren't going to return many of them
"""

# This text splitter is used to create the child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)


# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents",
    embedding_function=bge_embeddings  #OpenAIEmbeddings()
)

# The storage layer for the parent documents
store = InMemoryStore()

full_doc_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

full_doc_retriever.add_documents(docs, ids=None)

# our
list(store.yield_keys())

# vectorstore

sub_docs = vectorstore.similarity_search("what is langsmith", k=2)

len(sub_docs)

print(sub_docs[0].page_content)

retrieved_docs = full_doc_retriever.get_relevant_documents("what is langsmith")

len(retrieved_docs[0].page_content)

retrieved_docs[0].page_content

"""## Retrieving larger chunks
Sometimes, the full documents can be too big to want to retrieve them as is. In that case, what we really want to do is to first split the raw documents into larger chunks, and then split it into smaller chunks. We then index the smaller chunks, but on retrieval we retrieve the larger chunks (but still not the full documents).
"""

# This text splitter is used to create the parent documents - The big chunks
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# This text splitter is used to create the child documents - The small chunks
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="split_parents", embedding_function=bge_embeddings) #OpenAIEmbeddings()

# The storage layer for the parent documents
store = InMemoryStore()

big_chunks_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

big_chunks_retriever.add_documents(docs)

len(list(store.yield_keys()))

sub_docs = vectorstore.similarity_search("what is langsmith")

len(sub_docs)

print(sub_docs[0].page_content)

retrieved_docs = big_chunks_retriever.get_relevant_documents("what is langsmith")

len(retrieved_docs)

len(retrieved_docs[0].page_content)

print(retrieved_docs[0].page_content)

print(retrieved_docs[1].page_content)

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                 chain_type="stuff",
                                 retriever=big_chunks_retriever)

query = "What is Langsmith?"
qa.run(query)

