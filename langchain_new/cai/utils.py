import numpy as np
from langchain.chains import ConstitutionalChain, LLMChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder


def sts_eval(sent1: str, sent2: str):
    model = CrossEncoder('cross-encoder/stsb-distilroberta-base')
    sentence_combinations = [sent1, sent2]
    similarity_scores = model.predict([sentence_combinations])
    return similarity_scores


def cos_sim_eval(sent1: str, sent2: str):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Sentences are encoded by calling model.encode()
    emb1 = model.encode(sent1)
    emb2 = model.encode(sent2)
    cos_sim = util.cos_sim(emb1, emb2)
    return cos_sim.numpy()[0][0]


def get_cai_response(llm, sum_text):
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
    cai_response = constitutional_chain({"question:", sum_text})
    print("initial_response", cai_response['initial_output'], "\n")
    print("critique", cai_response['critiques_and_revisions'], "\n")
    print("output", cai_response['output'], "\n")
    return cai_response


def compare_sts_cos_cai_results(sum_text, cai_response):
    sts_score_input = sts_eval(sum_text, cai_response['output'])
    cos_sim_score_input = cos_sim_eval(sum_text, cai_response['output'])
    print("sts_score_input", sts_score_input[0])
    print("cos_sim_score_input", cos_sim_score_input)
    sts_score_output = sts_eval(cai_response['initial_output'], cai_response['output'])
    cos_sim_score_output = cos_sim_eval(cai_response['initial_output'], cai_response['output'])
    print("sts_score_output", sts_score_output[0])
    print("cos_sim_score_output", cos_sim_score_output)
    scores_arr = np.array([sts_score_input[0], cos_sim_score_input, sts_score_output[0], cos_sim_score_output])
    if np.all(scores_arr > 0.7, axis=0) > 0.8:
        return True
    else:
        return False
