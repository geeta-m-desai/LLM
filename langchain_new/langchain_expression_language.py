import os

print("Start")
os.environ["OPENAI_API_KEY"] = ""
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema.output_parser import StrOutputParser

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

model2 = OpenAI(
    model="text-davinci-003",
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
    "tell me an intersting fact about {subject}"
)

chain = prompt | model

chain.invoke({"subject": "Elvis"})

chain = prompt | model | StrOutputParser()

chain.invoke({"subject": "Elvis"})

chain = prompt | model2 | StrOutputParser()

chain.invoke({"subject": "Elvis"})

"""## Bindings"""

prompt = ChatPromptTemplate.from_template(
    "tell me 3 intersting facts about {subject}"
)

chain = prompt | model.bind(stop=["\n"]) | StrOutputParser()

chain.invoke({"subject": "Elvis"})

"""## Adding OpenAI Functions"""

functions = [
    {
        "name": "joke",
        "description": "A joke",
        "parameters": {
            "type": "object",
            "properties": {
                "setup": {
                    "type": "string",
                    "description": "The setup for the joke"
                },
                "punchline": {
                    "type": "string",
                    "description": "The punchline for the joke"
                }
            },
            "required": ["setup", "punchline"]
        }
    }
]
functions_chain = prompt | model.bind(function_call={"name": "joke"}, functions=functions)

functions_chain.invoke({"subject": "bears"}, config={})

"""### Functions Output Parser

"""

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

functions_chain = (
        prompt
        | model.bind(function_call={"name": "joke"}, functions=functions)
        | JsonOutputFunctionsParser()
)

response = functions_chain.invoke({"subject": "bears"})

response

response['punchline']

from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser

functions_chain = (
        prompt
        | model.bind(function_call={"name": "joke"}, functions=functions)
        | JsonKeyOutputFunctionsParser(key_name="setup")
)

functions_chain.invoke({"subject": "bears"})

"""## Retrievers"""

from langchain.schema.runnable import RunnablePassthrough
from operator import itemgetter

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough

# Create the retriever
fake_docs = ["James bond works for MI6", "Bond is a spy",
             "James Bond has a licence to kill", "James Bond likes cats"]
vectorstore = Chroma.from_texts(fake_docs, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)

chain.invoke("Who is James Bond?")

chain.invoke("What does James Bond like to do?")

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "language": itemgetter("language")
        } | prompt | model | StrOutputParser()

chain.invoke({"question": "where does James work?", "language": "english"})

chain.invoke({"question": "where does James work?", "language": "italian"})

"""## Tools"""

from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

template = """turn the following user input into a search query for a search engine:

{input}"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model | StrOutputParser() | search

print(chain.invoke({"input": "Who played james bond first"}))

chain = prompt | model | StrOutputParser()
print(chain.invoke({"input": "Who played james bond last"}))

"""## Arbitary Functions"""

from langchain.schema.runnable import RunnableLambda


def length_function(text):
    return len(text)


def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)


def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])


prompt = ChatPromptTemplate.from_template("what is {a} + {b}")

chain1 = prompt | model

chain = {
            "a": itemgetter("foo") | RunnableLambda(length_function),
            "b": {"text1": itemgetter("foo"), "text2": itemgetter("bar")} | RunnableLambda(multiple_length_function)
        } | prompt | model

chain.invoke({"foo": "bars", "bar": "gahs"})
