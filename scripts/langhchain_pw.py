if __name__ == "__main__":
    import pprint
    from pathlib import Path
    from typing import Annotated, Sequence

    import pandas as pd
    from dotenv import load_dotenv
    from langchain import hub
    from langchain.tools.retriever import create_retriever_tool
    from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
    from langchain_community.tools.playwright.utils import (
        create_sync_playwright_browser,
    )
    from langchain_core.messages import BaseMessage, HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import create_react_agent
    from playwright.sync_api import sync_playwright
    from pydantic import BaseModel, Field
    from typing_extensions import TypedDict

    # from rag import load_retriever
    # from utils.args import parse_args

    load_dotenv()
    # args = parse_args()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite"  # Most cost-efficient model
    )

    # retriever_tool = create_retriever_tool(
    #    load_retriever(args.reload_data),
    #    "candidate_data_retriever",
    #    "Contains candidate information, working & education experience, technical skills and qualifications",
    # )
    browser = create_sync_playwright_browser(False)
    pw_tools = PlayWrightBrowserToolkit.from_browser(sync_browser=browser).get_tools()

    graph = create_react_agent(llm, pw_tools)

    for s in graph.stream(
        {
            "messages": [
                (
                    "user",
                    f"Try apply for this job for the candidate: https://www.linkedin.com/jobs/view/4203903356. Try to extract the job description and find the apply button",
                )
            ]
        },
        stream_mode="values",
    ):
        message = s["messages"][-1]
        import pdb

        pdb.set_trace()
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
