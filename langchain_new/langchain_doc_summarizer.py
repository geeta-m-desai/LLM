import os, tempfile
import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

# Streamlit app
st.title('LangChain Doc Summarizer')

# Get OpenAI API key and source document input
openai_api_key = ""
source_doc = st.file_uploader("Upload Source Document", type="pdf")

# Check if the 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not openai_api_key.strip() or not source_doc:
        st.write(f"Please provide the missing fields.")
    else:
        try:
            # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(source_doc.read())
            loader = PyPDFLoader(tmp_file.name)
            pages = loader.load_and_split()
            #os.remove(tmp_file.name)

            # Create embeddings for the pages and insert into Chroma database
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectordb = Chroma.from_documents(pages, embeddings)
            query="Write a summary of the document. Provide list of research papers discussed in the document in a bullet list."
            vectordb.max_marginal_relevance_search(query, k=2)
            # Initialize the OpenAI module, load and run the summarize chain
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
            chain = load_summarize_chain(llm, chain_type="stuff")
            search = vectordb.similarity_search(" k=1 ")
            summary = chain.run(input_documents=search)

            st.write(summary)
        except Exception as e:
            st.write(f"An error occurred: {e.__cause__}")
