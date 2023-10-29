import os
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from datasets import load_dataset
from langchain_new.cai.utils import sts_eval, cos_sim_eval, get_cai_response, compare_sts_cos_cai_results
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0)


def run_llm_sum_cai(file_path):
    text_splitter = CharacterTextSplitter()
    textloader = TextLoader(file_path)
    docs1 = textloader.load_and_split(text_splitter)
    sum_chain = load_summarize_chain(llm, chain_type='stuff')
    sum_text = sum_chain.run(docs1)
    print("sum text ...", sum_text, "\n")
    cai_response = get_cai_response(llm, sum_text)
    if compare_sts_cos_cai_results(sum_text, cai_response):
        return cai_response["output"]
    else:
        return "Unexpected context generated. Please verify with human feedback"


if __name__ == "__main__":
    dataset = load_dataset("CarperAI/openai_summarize_tldr")
    print(type(dataset))
    print("Final outcome ",  run_llm_sum_cai("../xxx.txt"))
