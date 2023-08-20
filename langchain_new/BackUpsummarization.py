
db = Chroma(persist_directory="/Users/geetadesai/LLM", embedding_function=embeddings)
# Create embeddings for the pages and insert into Chroma database

vectordb = Chroma.from_documents(pages, embeddings, persist_directory="/Users/geetadesai/LLM/")

# Initialize the OpenAI module, load and run the summarize chain
llm = OpenAI(temperature=0)
chain = load_summarize_chain(llm, chain_type="stuff")
search = vectordb.similarity_search(" ")
# summary = chain.run(input_documents=search, question="Write a summary within 150 words.")
# print("vector summary", summary)
search1 = db.similarity_search(" ")
summary1 = chain.run(input_documents=search1, question="Write a summary within 150 words. You are legal advisor. Rephrase input for legal policies")
print("vector summary1", summary1)

print(chain.run(docs))
# output_summary = chain.run(pages)
# print("output summary", output_summary)
#
# just for creating the vector store. It can't actually be used as a retriever.
# index = VectorstoreIndexCreator(vectorstore_cls=Chroma, embedding=embeddings,
#                                 vectorstore_kwargs={"persist_directory": "/Users/geetadesai/LLM"}).from_loaders([loader])
