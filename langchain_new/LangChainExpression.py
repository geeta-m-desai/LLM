if __name__ == "__main__":
    print("Start...")
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate

    model = ChatOpenAI(openai_api_key="")
    prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
    chain = prompt | model

    print(chain.invoke({"foo": "bears"}))
    print("End...")
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.schema.runnable import RunnablePassthrough
    import chromadb
    chroma_client = chromadb.Client()
