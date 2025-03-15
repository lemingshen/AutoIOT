import config

from utils import prompt
from langchain import hub
from typing import List, Union
from langchain.agents import AgentExecutor
from langchain_core.documents import Document
from langchain.agents import create_openai_functions_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def load_new_webpage(web_paths: List[str]) -> List[Document]:
    print("We will further search information from these websites:")
    for URL in web_paths:
        print(URL)
    
    web_loader = WebBaseLoader(web_paths=web_paths)
    document_list = web_loader.load()
    
    return document_list


def create_context_retrieval_tool(document_list):
    vector = DocArrayInMemorySearch.from_documents(document_list, config.embeddings)
    retriever_tool = create_retriever_tool(
        retriever=vector.as_retriever(),
        name="context_document_search",
        description=prompt.context_prompt
    )
    
    return retriever_tool


def create_agent(have_message_history, tool_list, message_history):
    search_prompt = hub.pull("hwchase17/openai-functions-agent")
    
    agent = create_openai_functions_agent(config.llm, tool_list, search_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True)
    
    if have_message_history:
        return RunnableWithMessageHistory(
            agent_executor,
            lambda seesion_id: message_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
    else:
        return agent_executor


def sanitize_output(text: str):
    _, after = text.split("```python")
    return after.split("```")[0]