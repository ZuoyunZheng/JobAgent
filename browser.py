import asyncio

from browser_use import ActionResult, Agent, Browser, BrowserConfig, Controller
from browser_use.browser.context import BrowserContextConfig
from pypdf import PdfReader


async def main(llm, linkedin_url="https://www.linkedin.com/jobs/view/4199448826"):
    config = BrowserConfig(chrome_instance_path="/usr/bin/google-chrome-stable")
    browser = Browser(config=config)

    task = (
        f"Go to website {linkedin_url}."
        f"Find the button that is either 'Easy Apply' or 'Apply', click it to start the application process."
        "Leave the prefilled information as is."
        "Consult the candidate cv for context before filling each blank by forming assertive sentences."
        "If there are relevant context from candidate cv fill in accordingly, otherwise ask human for information."
        "Finally defer to human for approval before clicking next or sending the application"
    )
    initial_actions = [
        {"open_tab": {"url": linkedin_url}},
    ]

    controller = Controller()

    @controller.action("Ask user for information")
    def ask_human(question: str) -> str:
        answer = input(f"\n{question}\nInput: ")
        return ActionResult(extracted_content=answer)

    @controller.action("Defer to human input and wait for approval")
    def defer_human(question: str) -> str:
        input("Please check result so far and press Enter to continue...")
        return ActionResult(extracted_content="Approved, proceed")

    @controller.action("Read candidate cv for context to fill forms")
    def read_cv(input: str) -> str:
        pdf = PdfReader("data/ZuoyunZhengCV.pdf")
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        logger.info(f"Read cv with {len(text)} characters")
        return ActionResult(extracted_content=text, include_in_memory=True)

    agent = Agent(
        browser=browser,
        task=task,
        # initial_actions=initial_actions,
        llm=llm,
        use_vision=True,
        controller=controller,
    )
    await agent.run()


if __name__ == "__main__":
    import os
    import pprint
    from pathlib import Path
    from typing import Annotated, Sequence

    import pandas as pd
    from dotenv import load_dotenv
    from langchain import hub
    from langchain.tools.retriever import create_retriever_tool
    from langchain_core.messages import BaseMessage, HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from pydantic import BaseModel, Field, SecretStr
    from typing_extensions import TypedDict

    # from langgraph.prebuilt import create_react_agent
    from rag import load_retriever
    from utils.args import parse_args

    load_dotenv()
    args = parse_args()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", api_key=SecretStr(os.getenv("GEMINI_API_KEY"))
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
