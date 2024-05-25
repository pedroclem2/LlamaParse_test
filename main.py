import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, Settings
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser

import inquirer
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout

def load_api_keys():
    load_dotenv()
    llama_cloud_key = os.getenv("LLAMA_CLOUD_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not llama_cloud_key:
        raise KeyError("llama cloud api key not found in environment variables")
    if not openai_api_key:
        raise KeyError("openai api key not found in environment variables")

    return llama_cloud_key, openai_api_key

def initialize_llm():
    llm = OpenAI(model="gpt-4o")
    Settings.llm = llm
    return llm

def load_and_parse_documents(filepath, llm):
    documents = LlamaParse(result_type="markdown").load_data(filepath) #further explore 'parsing_instruction' parameter
    print(documents[0].text[0:1000])

    #element node parser for parsing the md output into a set of table and text nodes
    node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)
    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    return base_nodes, objects

def create_query_engine(base_nodes, objects):
    recursive_index = VectorStoreIndex(nodes=base_nodes + objects)
    query_engine = recursive_index.as_query_engine(similarity_top_k=25)
    return query_engine

def main():
    console = Console()
    layout = Layout()

    llama_cloud_key, openai_api_key = load_api_keys()
    llm = initialize_llm()
    base_nodes, objects = load_and_parse_documents("data/Merged Cell Table.pdf", llm)
    query_engine = create_query_engine(base_nodes, objects)

    questions = [
        inquirer.Text('search_query', message="Enter your query:")
    ]
    answers = inquirer.prompt(questions)
    search_query = answers['search_query']

    response = query_engine.query(search_query)

    # response_lines = response.count('\n') + 1
    # min_height = 10
    # panel_height = max(response_lines, min_height)

    layout.split_row(
        Layout(name="query"),
        Layout(name="response")
    )
    layout["query"].update(Panel(f"User query: {search_query}", style="bold cyan", height=10))
    layout["response"].update(Panel(f"{response}", title="Query Response", border_style="bold green", height=10))
    console.print(layout)

if __name__ == "__main__":
    main()
