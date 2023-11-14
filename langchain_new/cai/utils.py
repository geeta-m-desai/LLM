import csv
import warnings

import evaluate
import numpy as np
from datasets import load_dataset
from langchain.chains import ConstitutionalChain, LLMChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer, util

warnings.simplefilter('ignore')
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def calculate_sts_score(sent1: str, sent2: str):
    model = CrossEncoder('cross-encoder/stsb-distilroberta-base')
    sentence_combinations = [sent1, sent2]
    similarity_scores = model.predict([sentence_combinations])
    return similarity_scores


def calculate_cos_score(sent1: str, sent2: str):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Sentences are encoded by calling model.encode()
    emb1 = model.encode(sent1)
    emb2 = model.encode(sent2)
    cos_sim = util.cos_sim(emb1, emb2)
    return cos_sim.numpy()[0][0]


def get_cai_response(llm, sum_text):
    qa_prompt = PromptTemplate(
        template="""You are ethical and legal advisor. 
        Provide advice only if the input is related to ethical and legal matters. 
        Do not add any explanation. 
        Provide only facts. 
    Question: {input}

    QA answer:""",
        input_variables=["input"],
    )

    qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

    ethical_principle = ConstitutionalPrinciple(
        name="Ethical Principle",
        critique_request="""The model should only talk about ethical and legal things. 
        Mark critique as No if input ethical and legal.
        Mark critique as needed as Yes if input is unethical or illegal and provide 2-3 reasons behind marking critique as needed
        Do not add any explanation. 
        Provide only facts.""",
        revision_request="""If critique is needed then Rewrite the input to be both ethical and legal based on reasons provided in critique.Do not add any explanation. 
        Provide only facts based on input.""",
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
    sts_score_input = calculate_sts_score(sum_text, cai_response['output'])
    cos_sim_score_input = calculate_cos_score(sum_text, cai_response['output'])
    print("sts_score_input", sts_score_input[0])
    print("cos_sim_score_input", cos_sim_score_input)
    sts_score_output = calculate_sts_score(cai_response['initial_output'], cai_response['output'])
    cos_sim_score_output = calculate_cos_score(cai_response['initial_output'], cai_response['output'])
    print("sts_score_output", sts_score_output[0])
    print("cos_sim_score_output", cos_sim_score_output)
    scores_arr = np.array([sts_score_input[0], cos_sim_score_input, sts_score_output[0], cos_sim_score_output])
    if calculate_cos_score(cai_response['critiques_and_revisions'][0][0],
                           "No critique needed.") > 0.9 and calculate_sts_score(
        cai_response['critiques_and_revisions'][0][0], "No critique needed."):
        return sum_text
    else:
        if np.all(scores_arr > 0.7, axis=0) > 0.8:
            return cai_response['output']
        else:
            return "Unexpected context generated. Please verify with human feedback"


def compare_sts_cos_cai_rag_results(orig_text, query, result_text, cai_response, type):
    sts_score_output = calculate_sts_score(cai_response['initial_output'], cai_response['output'])
    cos_sim_score_output = calculate_cos_score(cai_response['initial_output'], cai_response['output'])
    # blue_score = calculate_blue_score([cai_response['initial_output']], [cai_response['output']])['bleu']
    rouge_score = calculate_rouge_score([cai_response['initial_output']], [cai_response['output']])['rougeL']
    bert_score = calculate_bert_score([cai_response['initial_output']], [cai_response['output']])['recall'][0]
    print("sts_score_output", sts_score_output[0])
    print("cos_sim_score_output", cos_sim_score_output)
    # print("blue_score", blue_score)
    print("rouge_score", rouge_score)
    print("bert_score", bert_score)
    scores_arr = np.array([sts_score_output[0], cos_sim_score_output, rouge_score, bert_score])
    if calculate_cos_score(cai_response['critiques_and_revisions'][0][0],
                           "No critique needed.") > 0.9 and calculate_sts_score(
        cai_response['critiques_and_revisions'][0][0], "No critique needed."):
        cai_text = result_text
    else:
        if np.all(scores_arr > 0.7, axis=0) > 0.8:
            cai_text = cai_response['output']
        else:
            cai_text = cai_response['output'] + "\n \n Unexpected content generated. Please verify with human feedback"
    write_file_rag_results(orig_text, query, result_text, cai_text,
                           sts_score_output[0], cos_sim_score_output, rouge_score, bert_score, type)


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


def write_file_rag_results(orig_text, query, qa_result_text, cai_text,
                           sts_score_output,
                           cos_sim_score_output, rouge_score_output, bert_score_output, type):
    header = ['orig_text', 'query', 'qa_result_text', 'cai_sum_text',
              'sts_score_output',
              'cos_sim_score_output', 'rouge_score_output', 'bert_score_output', 'type']
    data = [orig_text, query, qa_result_text, cai_text, sts_score_output,
            cos_sim_score_output, rouge_score_output, bert_score_output, type]

    file_name = 'rag_output' + '.csv'
    if not os.path.exists(file_name):
        print("No File")
        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(data)
    else:
        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
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


def query_llm(llm, query, context):
    qa_prompt = PromptTemplate(
        template="""You are advisor. Provide response based on context {context}. Provide only facts. Do not make up answers.
         Do not add any explanation.
        Question: {query}

        QA answer:""",
        input_variables=["query","context"],
    )

    qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
    return qa_chain({"query":query,"context":context})


def chain_of_verification(llm, input, context, response):
    qa_prompt = PromptTemplate(
        template="""Based on the response provided {response} for context {context}, suggest 2-3 questions to verify key facts that could identify inaccuracies in the response if any.""",
        input_variables=["context", "response"],
    )

    qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
    verification = (qa_chain({'context': context, 'response': response}))
    print("verification ", verification['text'], "\n")
    lst = verification['text'].split("?")
    lst_answers = {}
    for i in lst:
        print("quest *** ", i, "\n")
        if i != '':
            res = query_llm(llm, i, context)['text']
            print("res", res, "\n")
            if res is not None:
                lst_answers[i] = res
    re_rerun_response = "Initial response: " + response + " Verification questions: "
    for quest in lst_answers.keys():
        re_rerun_response = (re_rerun_response + quest +
                             "? " + lst_answers[quest])
    re_rerun_response = (re_rerun_response + input + "?")
    print(re_rerun_response)
    final_res = query_llm(llm, re_rerun_response,context)
    print("final_res", final_res)
    return final_res


if __name__ == "__main__":
    # predictions = [
    #     " 30 years old weighed themselves weekly and measured themselves monthly in order to track their progress in their weight loss journey. They had recently hit a plateau of 222 pounds and felt disappointed when they weighed themselves on Monday and saw no progress. However, when they measured themselves on their measure-in day, they discovered that they had lost a total 8 inches from their starting point on 12/23/14. This was a cause for celebration, as they were now the lightest and smallest they had been since right around high school."]
    # references = [
    #     " 30 years old weighed themselves weekly and measured themselves monthly in order to track their progress in their weight loss journey. They had recently hit a plateau of 222 pounds and felt disappointed when they weighed themselves on Monday and saw no progress. However, when they measured themselves on their measure-in day, they discovered that they had lost a total 8 inches from their starting point on 12/23/14."]
    # print(calculate_rouge_score(predictions, references))
    # print(calculate_blue_score(predictions, references))
    # print(calculate_bert_score(predictions, references))
    # print(calculate_sts_score(predictions[0], references[0]))
    # print(calculate_cos_score(predictions[0], references[0]))
    llm = OpenAI()
    dataset = load_dataset("CarperAI/openai_summarize_tldr")
    train_dataset = dataset.data['train']
    test_dataset = dataset.data['test']
    valid_dataset = dataset.data['valid']
    docs = train_dataset[0][1]
    docs = docs.as_py()
    response = "Since this context information does not contain any information on the president of Bhutan, it is not possible to answer this query."

    chain_of_verification(llm, "How will be the president of Bhutan?", docs, response)
