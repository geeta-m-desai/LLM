
# !pip -q install langchain huggingface_hub openai tiktoken pypdf
# !pip -q install google-generativeai chromadb unstructured
# !pip -q install sentence_transformers
# !pip -q install -U FlagEmbedding

"""### Download the Data & Utils"""

import os
import requests
import zipfile
from io import BytesIO
import textwrap

def download_and_extract_zip(url, target_folder):
    # Ensure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Download the file from the URL
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {url}")

    # Unzip the file in memory
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(target_folder)

    print(f"Files extracted to {target_folder}")


def zip_folder(folder_path, zip_file_path):
    # Create a ZipFile object in write mode
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the directory
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Create a path relative to the folder to avoid storing absolute paths
                relative_path = os.path.relpath(os.path.join(root, file), os.path.dirname(folder_path))
                # Add file to the zip file
                zipf.write(os.path.join(root, file), arcname=relative_path)

    print(f"{zip_file_path} created successfully.")



def wrap_text(text, width=90): #preserve_newlines
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

# URL of the zip file
url = "https://www.dropbox.com/scl/fi/av3nw07o5mo29cjokyp41/singapore_text_files_languages.zip?rlkey=xqdy5f1modtbnrzzga9024jyw&dl=1" # Ensure dl=1 for direct download

# Folder to save extracted files
folder = "singapore_text"

# Call the function
download_and_extract_zip(url, folder)

## download the chroma DB
url = 'https://www.dropbox.com/scl/fi/3kep8mo77h642kvpum2p7/singapore_chroma_db.zip?rlkey=4ry4rtmeqdcixjzxobtmaajzo&dl=1'
download_and_extract_zip(url, '.')

import os




"""## Google

## Imports
"""

from langchain.llms import GooglePalm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
import langchain

"""## Load in Docs"""

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader

# Commented out IPython magic to ensure Python compatibility.
# %%time
loader = DirectoryLoader('/content/singapore_text/Textfiles3/English/', glob="*.txt", show_progress=True)
docs = loader.load()

len(docs)
# docs = docs[:10]
len(docs)

docs[0]

raw_text = ''
for i, doc in enumerate(docs):
    text = doc.page_content
    if text:
        raw_text += text

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 100,
    length_function = len,
    is_separator_regex = False,
)

texts = text_splitter.split_text(raw_text)

len(texts)

texts[1]

"""## BGE Embeddings"""

from langchain.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-small-en-v1.5"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

embedding_function = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'},
    encode_kwargs=encode_kwargs,

)

"""## Vector DB"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# ### Make the chroma and persiste to disk
# # db = Chroma.from_texts(texts,
# #                        embedding_function,
# #                        persist_directory="./chroma_db")
# 
# 
# 
# ### load from disk
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
# 
# 
# 
#

### Save to zip
# zip_folder('/content/chroma_db', 'chroma_db.zip')

query = "Tell me about Universal Studios Singapore?"

db.similarity_search(query, k=5)

"""## Setup a Retriever"""

retriever = db.as_retriever(k=5) # can add mmr fetch_k=20, search_type="mmr"

retriever.get_relevant_documents(query)[1]

"""## Chat chain"""

from operator import itemgetter

from langchain.chat_models import ChatGooglePalm

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatGooglePalm()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

text_reply = chain.invoke("Tell me about Universal Studio Singapore")

print(wrap_text(text_reply))

"""## With RagFusion"""

from langchain.chat_models import ChatGooglePalm
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import ChatMessagePromptTemplate, PromptTemplate

prompt = ChatPromptTemplate(input_variables=['original_query'], messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant that generates multiple search queries based on a single input query.')), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['original_query'], template='Generate multiple search queries related to: {question} \n OUTPUT (4 queries):'))])

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant that generates multiple search queries based on a single input query."),
#     ("user", "Generate multiple search queries related to: {question}/n OUTPUT (4 queries):"),
# ])

prompt

generate_queries = (
    prompt | ChatGooglePalm(temperature=0) | StrOutputParser() | (lambda x: x.split("\n"))
)

original_query = "universal studios Singapore"

from langchain.load import dumps, loads


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

ragfusion_chain = generate_queries | retriever.map() | reciprocal_rank_fusion

langchain.debug = True

ragfusion_chain.input_schema.schema()

ragfusion_chain.invoke({"question": original_query})

from langchain.schema.runnable import RunnablePassthrough
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

full_rag_fusion_chain = (
    {
        "context": ragfusion_chain,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

full_rag_fusion_chain.input_schema.schema()

full_rag_fusion_chain.invoke({"question": "Tell me about Universal Studio Singapore?"})
