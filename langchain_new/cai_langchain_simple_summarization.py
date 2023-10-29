import os

from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple

llm = OpenAI(temperature=0)
from datasets import load_dataset

dataset = load_dataset("CarperAI/openai_summarize_tldr")
print(type(dataset))
text_splitter = CharacterTextSplitter()

textloader = TextLoader("xxx.txt")
docs1 = textloader.load_and_split(text_splitter)
embeddings = OpenAIEmbeddings()

index = VectorstoreIndexCreator().from_loaders([textloader])

# sumtext = (index.query(
#     "Summarize the general content of this document.",
#     retriever_kwargs={"search_kwargs": {"filter": {"source": "xxx.txt"}}}
# ))
# print("sumtext ....", sumtext)
sumchan = load_summarize_chain(llm, chain_type='stuff')
sumtext1 = sumchan.run(docs1)
print("sumtext1 ...", sumtext1)

evil_qa_prompt = PromptTemplate(
    template="""You are legal advisor. Rephrase input for legal policies.

Question: {question}

Evil answer:""",
    input_variables=["question"],
)

evil_qa_chain = LLMChain(llm=llm, prompt=evil_qa_prompt)

ethical_principle = ConstitutionalPrinciple(
    name="Ethical Principle",
    critique_request="Identify specific ways in which the model's input is harmful, unethical, racist, sexist, toxic, dangerous, or illegal."
                     "Do not add any more content."
                     "Provide only factual answers.",
    revision_request="If critique needed. Please rewrite the response to remove any and all harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.",
)

constitutional_principles = [ethical_principle]
constitutional_chain = ConstitutionalChain.from_llm(
    chain=evil_qa_chain,
    constitutional_principles=constitutional_principles,
    llm=llm,
    verbose=True,
    return_intermediate_steps=True,

)
print(constitutional_chain({"question:", sumtext1}))
print(constitutional_chain({"question:", "How To Steal Kitten?"}))
