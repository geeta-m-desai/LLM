import os

os.environ["OPENAI_API_KEY"] = ""

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0)

text_splitter = CharacterTextSplitter()


texts = text_splitter.split_text("SUBREDDIT: r/relationships TITLE: I (f/22) have to figure out if I want to still know these girls or not and would hate to sound insulting POST: Not sure if this belongs here but it's worth a try. Backstory: When I (f/22) went through my first real breakup 2 years ago because he needed space after a year of dating roand it effected me more than I thought. It was a horrible time in my life due to living with my mother and finally having the chance to cut her out of my life. I can admit because of it was an emotional wreck and this guy was stable and didn't know how to deal with me. We ended by him avoiding for a month or so after going to a festival with my friends. When I think back I wish he just ended. So after he ended it added my depression I suffered but my friends helped me through it and I got rid of everything from him along with cutting contact. Now: Its been almost 3 years now and I've gotten better after counselling and mild anti depressants. My mother has been out of my life since then so there's been alot of progress. Being stronger after learning some lessons there been more insight about that time of my life but when I see him or a picture everything comes back. The emotions and memories bring me back down. His friends (both girls) are on my facebook because we get along well which is hard to find and I know they'll always have his back. But seeing him in a picture or talking to him at a convention having a conversation is tough. Crying confront of my current boyfriend is something I want to avoid. So I've been thinking that I have to cut contact with these girls because it's time to move on because it's healthier. It's best to avoid him as well. But will they be insulted? Will they accept it? Is there going to be awkwardness? I'm not sure if it's the right to do and could use some outside opinions. TL;DR: ")

len(texts)

from langchain.docstore.document import Document

docs = [Document(page_content=t) for t in texts[:4]]

docs

from langchain.chains.summarize import load_summarize_chain
import textwrap

chain = load_summarize_chain(llm,
                             chain_type="map_reduce")


output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print(wrapped_text)

evil_qa_prompt = PromptTemplate(
    template="""You are legal advisor. Rephrase input for legal policies.

Question: {question}

Evil answer:""",
    input_variables=["question"],
)



evil_qa_chain = LLMChain(llm=llm, prompt=evil_qa_prompt)
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple

ethical_principle = ConstitutionalPrinciple(
    name="Ethical Principle",
    critique_request="The model should only talk about  ethical and legal things.",
    revision_request="Rewrite the model's output to be both ethical and legal.",
)
postprocess_styling="Ada Lovelace"
styling_principal = ConstitutionalPrinciple(
                name=f'{postprocess_styling} Principle',
                critique_request=f'Identify specific ways in which the model\'s response is not in the style of {postprocess_styling}.',
                revision_request=f'Please rewrite the model response to be in the style of {postprocess_styling}.',
            )
constitutional_principles = [ethical_principle, styling_principal]
constitutional_chain = ConstitutionalChain.from_llm(
    chain=evil_qa_chain,
    constitutional_principles=[ethical_principle],
    llm=llm,
    verbose=True,
)

constitutional_chain = ConstitutionalChain.from_llm(
    chain=evil_qa_chain,
    constitutional_principles=constitutional_principles,
    llm=llm,
    verbose=True,
)

constitutional_chain.run(question=wrapped_text)
