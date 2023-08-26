import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = ""

from langchain import OpenAI, PromptTemplate, LLMChain, HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple

# llm = HuggingFacePipeline.from_model_id(
#     model_id="/Users/geetadesai/LLaMA-2-7B-32K",
#     task="text-generation",
#     model_kwargs={"temperature": 0, "max_length": 200},
# )
llm = OpenAI(temperature=0)

text_splitter = CharacterTextSplitter()
text = "Officers were called to Lowe Street in Whitmore Reans, Wolverhampton at 17:00 GMT on Thursday when a 17-year-old boy was found with stab wounds. Several minutes later a second call was made to police when a shotgun was fired twice in nearby Deveron Close. West Midlands Police believe the men, who have now been bailed, were linked to both incidents. The teenager remains in hospital in a stable condition, police said. DCI Chris Hanson said: We believe both of these offences were linked and were the result of a dispute between two groups. The shooting happened following an argument between a group and a lone man with a gun."

texts = text_splitter.split_text(text)
textloader = TextLoader("xxx.txt")

from langchain.docstore.document import Document

docs = [Document(page_content=t) for t in texts[:1]]

embeddings = OpenAIEmbeddings()

index = VectorstoreIndexCreator().from_loaders([textloader])

sumtext = (index.query(
    "Summarize the general content of this document.",
    retriever_kwargs={"search_kwargs": {"filter": {"source": "xxx.txt"}}}
))
print("sumtext ....", sumtext)
sumchan = load_summarize_chain(llm, chain_type='stuff')
sumtext1 = sumchan.run(docs)
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
    critique_request="The model should only talk about  ethical and legal things.",
    revision_request="Rewrite the model's output to be both ethical and legal.",
)
postprocess_styling = "Ada Lovelace"
styling_principal = ConstitutionalPrinciple(
    name=f'{postprocess_styling} Principle',
    critique_request=f'Identify specific ways in which the model\'s response is not in the style of {postprocess_styling}.',
    revision_request=f'Please rewrite the model response to be in the style of {postprocess_styling}.',
)
constitutional_principles = [ethical_principle]
constitutional_chain = ConstitutionalChain.from_llm(
    chain=evil_qa_chain,
    constitutional_principles=constitutional_principles,
    llm=llm,
    verbose=True,
)
print(constitutional_chain.run(question=sumtext))
print(constitutional_chain.run(question=sumtext1))
