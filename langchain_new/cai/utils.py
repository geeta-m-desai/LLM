import csv

import evaluate
import numpy as np
from langchain.chains import ConstitutionalChain, LLMChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer, util


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
        template="""You are advisor. Provide response based on context.

    Question: {question}

    QA answer:""",
        input_variables=["question"],
    )

    qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

    ethical_principle = ConstitutionalPrinciple(
        name="Ethical Principle",
        critique_request="The model should only talk about ethical things. Mark Critique needed as Yes if input is not ethical",
        revision_request="If critique is needed then Rewrite the model's input to be both ethical. Do not add any explanation. Provide only facts.",
    )

    constitutional_principles = [ethical_principle]
    constitutional_chain = ConstitutionalChain.from_llm(
        chain=qa_chain,
        constitutional_principles=constitutional_principles,
        llm=llm,
        verbose=True,
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
    if cos_sim_eval(cai_response['critiques_and_revisions'][0][0], "No critique needed.") > 0.9 and sts_eval(
            cai_response['critiques_and_revisions'][0][0], "No critique needed."):
        return sum_text
    else:
        if np.all(scores_arr > 0.7, axis=0) > 0.8:
            return cai_response['output']
        else:
            return "Unexpected context generated. Please verify with human feedback"


def compare_sts_cos_cai_rag_results(orig_text, query, result_text, cai_response, type):
    sts_score_input = sts_eval(result_text, cai_response['output'])
    cos_sim_score_input = cos_sim_eval(result_text, cai_response['output'])
    print("sts_score_input", sts_score_input[0])
    print("cos_sim_score_input", cos_sim_score_input)
    sts_score_output = sts_eval(cai_response['initial_output'], cai_response['output'])
    cos_sim_score_output = cos_sim_eval(cai_response['initial_output'], cai_response['output'])
    print("sts_score_output", sts_score_output[0])
    print("cos_sim_score_output", cos_sim_score_output)
    scores_arr = np.array([sts_score_input[0], cos_sim_score_input, sts_score_output[0], cos_sim_score_output])
    if cos_sim_eval(cai_response['critiques_and_revisions'][0][0], "No critique needed.") > 0.9 and sts_eval(
            cai_response['critiques_and_revisions'][0][0], "No critique needed."):
        cai_text = result_text
    else:
        if np.all(scores_arr > 0.7, axis=0) > 0.8:
            cai_text = cai_response['output']
        else:
            cai_text = "Unexpected context generated. Please verify with human feedback"
    write_file_rag_results(orig_text, query, result_text, cai_text, sts_score_input[0], cos_sim_score_input,
                           sts_score_output[0], cos_sim_score_output, type)


def write_file(orig_text, sum_text, cai_text):
    data = [
        ['orig_text', 'sum_text', 'cai_sum_text'],
        [orig_text, sum_text, cai_text]
    ]
    file_name = 'output.csv'
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
        print("file Created")


def write_file_rag_results(orig_text, query, qa_result_text, cai_text, sts_score_input, cos_sim_score_input,
                           sts_score_output,
                           cos_sim_score_output, type):
    data = [
        ['orig_text', 'query', 'qa_result_text', 'cai_sum_text', 'sts_score_input', 'cos_sim_score_input',
         'sts_score_output',
         'cos_sim_score_output'],
        [orig_text, query, qa_result_text, cai_text, sts_score_input, cos_sim_score_input, sts_score_output,
         cos_sim_score_output]
    ]
    file_name = 'rag_output_' + type + '.csv'
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
        print("file Created")


# Recall based and compares N gram overlap with reference
def calculate_rouge_score(input_predictions, input_references):
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=input_predictions,
                            references=input_references)
    return results


# BLEU compares overlap in tokens from the predictions and references, instead of comparing meaning. This can lead to discrepancies between BLEU scores and human ratings.Shorter predicted translations achieve higher scores than longer ones, simply due to how the score is calculated. A brevity penalty is introduced to attempt to counteract this.
def calculate_blue_score(input_predictions, input_references):
    rouge = evaluate.load('bleu')
    results = rouge.compute(predictions=input_predictions,
                            references=input_references)
    return results


# BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference sentences by cosine similarity. It has been shown to correlate with human judgment on sentence-level and system-level evaluation. Moreover, BERTScore computes precision, recall, and F1 measure, which can be useful for evaluating different language generation tasks.
def calculate_bert_score(input_predictions, input_references):
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=input_predictions, references=input_references, lang="en")
    return results


if __name__ == "__main__":
    predictions = ["hello there", "general kenobi"]
    references = ["hello there", "general kenobi"]
    print(calculate_rouge_score(predictions, references))
    print(calculate_blue_score(predictions, references))
    print(calculate_bert_score(predictions, references))
