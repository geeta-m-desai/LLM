import os

from datasets import load_dataset
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_new.cai.utils import get_cai_response, compare_sts_cos_cai_results, write_file, calculate_rouge_score

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = OpenAI(temperature=0)


def run_llm_sum_cai(text):
    # text_splitter = CharacterTextSplitter()
    # textloader = TextLoader(file_path)
    # docs1 = textloader.load_and_split(text_splitter)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # doc = Document(page_content=documents, metadata={"source": "local"})
    texts = text_splitter.split_text(text)
    docs1 = [Document(page_content=t) for t in texts[:4]]
    sum_chain = load_summarize_chain(llm, chain_type='stuff')
    sum_text = sum_chain.run(docs1)
    print("sum text ...", sum_text, "\n")
    cai_response = get_cai_response(llm, sum_text)
    final_cai_outcome = compare_sts_cos_cai_results(sum_text, cai_response)
    write_file(texts, sum_text, final_cai_outcome)
    # calculate_rouge(sum_text, final_cai_outcome)
    # return compare_sts_cos_cai_results(sum_text, cai_response)


if __name__ == "__main__":
    dataset = load_dataset("CarperAI/openai_summarize_tldr")
    train_dataset = dataset.data['train']
    test_dataset = dataset.data['test']
    valid_dataset = dataset.data['valid']
    docs = train_dataset[0][1]
    docs = docs.as_py()
    run_llm_sum_cai(docs)
