from browser_use import Agent, Browser, BrowserConfig, Controller
from browser_use.browser.context import BrowserContextConfig
import asyncio

async def main(llm):
    config = BrowserConfig(chrome_instance_path="/usr/bin/google-chrome-stable")
    browser = Browser(config=config)
    task = f"On the url https://www.linkedin.com/jobs/view/4199448826, find the blue Easy Apply button, click it and start the application process, fill out as much as you can and finally defer to human for approval"
    controller = Controller()
    
    @controller.action("Defer to human input and wait for approval")
    def defer_human():
        input("Please check result so far and press Enter to continue...")
        return

    agent = Agent(
        browser=browser,
        task=task,
        llm=llm,
        controller=controller,
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

    asyncio.run(main(llm))
