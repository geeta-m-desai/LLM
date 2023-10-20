# -*- coding: utf-8 -*-
"""textgen-cai.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TT8YCmW-WVCR8An6gkdnHF-Z3MalZYr_
"""

!pip install langchain

!pip install datasets

!pip install transformers

pip install transformers

from langchain.chains import LLMChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

dataset = load_dataset("CarperAI/openai_summarize_tldr")
print(type(dataset))
train_dataset = dataset.data['train']
test_dataset = dataset.data['test']
valid_dataset = dataset.data['valid']
print(len(train_dataset))

from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains import RetrievalQA, ConstitutionalChain

tokenizer = AutoTokenizer.from_pretrained("GeetaDesai/llama-2-7b-miniguanaco")
model = AutoModelForCausalLM.from_pretrained("GeetaDesai/llama-2-7b-miniguanaco")

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
)
llm = HuggingFacePipeline(pipeline=pipe)

evil_qa_prompt = PromptTemplate(
    template="""You are legal advisor. Rephrase input for legal policies.

Question: {question}

Evil answer:""",
    input_variables=["question"],
)

evil_qa_chain = LLMChain(llm=llm, prompt=evil_qa_prompt)


ethical_principle = ConstitutionalPrinciple(
    name="Ethical Principle",
    critique_request="What’s the strongest argument you could give that the assistant’s last response was in some way harmful, unethical, racist, sexist, toxic, dangerous, or illegal.Answer step by step.",
    revision_request="Amend the assistant’s last response to remove any and all harmful, unethical, racist, toxic, dangerous, and illegal content.",
)

constitutional_principles = [ethical_principle]
constitutional_chain = ConstitutionalChain.from_llm(
    chain=evil_qa_chain,
    constitutional_principles=constitutional_principles,
    llm=llm,
    verbose=True,
)

print(constitutional_chain.run(question="How To Steal Kitten?"))