import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser

def load_api_keys():
    load_dotenv()
    llama_cloud_key = os.getenv("LLAMA_CLOUD_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not llama_cloud_key:
        raise KeyError("llama cloud api key not found in environment variables")
    if not openai_api_key:
        raise KeyError("openai api key not found in environment variables")
    
    return llama_cloud_key, openai_api_key

def initialize_llm_and_embeddings():
    embeddings = OpenAIEmbedding(model="text-embedding-3-small")
    llm = OpenAI(model="gpt-4o")
    Settings.llm = llm
    return llm, embeddings

def load_and_parse_documents(filepath, llm):
    documents = LlamaParse(result_type="markdown").load_data(filepath)
    print(documents[0].text[0:1000])
    
    node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)
    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    return base_nodes, objects

def create_query_engine(base_nodes, objects):
    recursive_index = VectorStoreIndex(nodes=base_nodes + objects)
    query_engine = recursive_index.as_query_engine(similarity_top_k=25)
    return query_engine

def main():

    llama_cloud_key, openai_api_key = load_api_keys()
    llm, embeddings = initialize_llm_and_embeddings()
    base_nodes, objects = load_and_parse_documents("data/article.pdf", llm)
    query_engine = create_query_engine(base_nodes, objects)
    query = "Who are the authors of the article?"
    response = query_engine.query(query)
    print(response)

if __name__ == "__main__":
    main()
