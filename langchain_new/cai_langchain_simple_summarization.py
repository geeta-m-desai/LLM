import os

from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
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
sumchain = load_summarize_chain(llm, chain_type='stuff')
sumtext1 = sumchain.run(docs1)
print("sumtext1 ...", sumtext1, "\n")

qa_prompt = PromptTemplate(
    template="""You are legal advisor. Rephrase input for legal policies.

Question: {question}

QA answer:""",
    input_variables=["question"],
)

qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

ethical_principle = ConstitutionalPrinciple(
    name="Ethical Principle",
    critique_request="Identify specific ways in which the model's input is harmful, unethical, racist, sexist, toxic, dangerous, or illegal."
                     "Do not add any more content."
                     "Provide only factual answers.",
    revision_request="If critique needed. Please rewrite the response to remove any and all harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.",
)

constitutional_principles = [ethical_principle]
constitutional_chain = ConstitutionalChain.from_llm(
    chain=qa_chain,
    constitutional_principles=constitutional_principles,
    llm=llm,
    verbose=False,
    return_intermediate_steps=True,

)
cai_response = constitutional_chain({"question:", sumtext1})
print("initial_response", cai_response['initial_output'],"\n")
print("critique", cai_response['critiques_and_revisions'], "\n")
print("output", cai_response['output'], "\n")
# print(constitutional_chain({"question:", "How To Steal Kitten?"}))
model = CrossEncoder('cross-encoder/stsb-distilroberta-base')
sentence_combinations= [cai_response['initial_output'],cai_response['output']]
similarity_scores = model.predict(sentence_combinations)
print(similarity_scores)
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

#Sentences are encoded by calling model.encode()
emb1 = model.encode(sentence_combinations)


cos_sim = util.cos_sim(emb1[0],emb1[1])
print("Cosine-Similarity:", cos_sim.numpy()[0][0])
