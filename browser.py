from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig
import asyncio

async def main(llm):
    config = BrowserConfig(chrome_instance_path="/usr/bin/google-chrome-stable")
    browser = Browser(config=config)
    task = f"Try apply for this job for the candidate: https://www.linkedin.com/jobs/view/4203903356. Avoid logging in to LinkedIn if you can and try to navigato the company's website to apply by clicking the apply button on the given linkedin link"
    agent = Agent(
        browser=browser,
        task=task,
        llm=llm,
    )
    await agent.run()

if __name__ == "__main__":
    import pprint
    from pathlib import Path
    from typing import Annotated, Sequence

    import pandas as pd
    from dotenv import load_dotenv
    from langchain import hub
    from langchain.tools.retriever import create_retriever_tool
    from langchain_core.messages import BaseMessage, HumanMessage
    from langchain_ollama import ChatOllama
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from pydantic import BaseModel, Field
    from typing_extensions import TypedDict

    # from langgraph.prebuilt import create_react_agent
    from rag import load_retriever
    from utils.args import parse_args

    load_dotenv()
    args = parse_args()

    llm = ChatOllama(
        model="qwen2.5:1.5b",
    )
    retriever = load_retriever(args.reload_data)

    retriever_llm = llm.bind_tools(
        [
            create_retriever_tool(
                retriever,
                "candidate_data_retriever",
                "Contains candidate information, working & education experience, technical skills and qualifications",
            )
        ]
    )

    asyncio.run(main(retriever_llm))
