import pandas
from jobspy import scrape_jobs


def jobspy_scrape_jobs(
    site_names: list[str], terms: str, location: str, num_results: int, time_limit: int
) -> pandas.DataFrame:
    """
    Scrape jobs from job boards such as LinkedIn and glassdoor

    Args:
        site_names (list): List of job board sites to scrape from, available sites are linkedin and glassdoor
        terms (list): List of job titles to search for
        location (str): Location to search for jobs
        num_results (int): Number of results to return
        time_limit (int): Time limit in hours to scrape for jobs
    """
    return scrape_jobs(
        site_name=site_names,
        search_term=terms,
        location=location,
        results_wanted=num_results,
        hours_old=time_limit,
        linkedin_fetch_description=True,
    )


if __name__ == "__main__":
    import pprint
    from pathlib import Path
    from typing import Annotated, Sequence

    import pandas as pd
    from dotenv import load_dotenv
    from langchain import hub
    from langchain_core.messages import BaseMessage, HumanMessage
    from langchain_ollama import ChatOllama
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from pydantic import BaseModel, Field
    from typing_extensions import TypedDict

    from rag import load_retriever
    from utils.args import parse_args

    load_dotenv()
    args = parse_args()

    # LLM model
    llm = ChatOllama(
        model="qwen2.5:1.5b",
    )
    retriever = load_retriever(args.reload_data)

    # ---- RAG ---- #

    # State
    class State(TypedDict):
        jobs: list

    # Node
    # Bind model with tools
    # Jobspy search tool node without llm input
    def scrape_linkedin(state):
        if Path("jobs.pkl").exists():
            jobs = pd.read_pickle("jobs.pkl")
        else:
            jobs = jobspy_scrape_jobs(
                "linkedin", "Machine Learning Engineer", "Germany", 5, 72
            )
            jobs.to_pickle("jobs.pkl")
        return {"jobs": jobs.values.tolist()}

    def filter_jobs(state):
        jobs = []

        # 1. Prompt
        rag_prompt = hub.pull("rlm/rag-prompt")

        # 2. Structured LLM
        class fit(BaseModel):
            """Binary score for relevance check."""

            fit_value: str = Field(description="Relevance score 'yes', 'maybe' or 'no'")
            reason: str = Field(description="Detailed reason for fit")

        structured_llm = llm.with_structured_output(fit)

        for job in state["jobs"]:
            # 3. Summarize criteria from job
            # company, title, description
            print(f"Filtering {job[5]} ({job[4]}): {job[19][:250]}")
            # TODO: 1. Summarize description 2. due-diligence on the company
            response = llm.invoke(
                f"What are the main qualification and skills required for the position of {job[4]} at {job[5]} with the description {job[19][:1000]}"
            )

            # 4. Retrieve candidate qualification
            docs = retriever.invoke(response.content)

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            context = format_docs(docs)

            question = f"Is the candidate fit for the following position:\n\n {response}, give a relevance score 'yes' for a great fit, 'maybe' for multiple matching skills and qualifications or 'no' for no match in skills and interest"
            chain = rag_prompt | structured_llm
            response = chain.invoke({"context": context, "question": question})

            if response.fit_value in ["yes", "maybe"]:
                print(f"Relevant job: {response.reason}")
                jobs.append(job)
            else:
                print(f"Irrelevant job: {response.reason}")
        return {"jobs": jobs}

    workflow = StateGraph(State)

    workflow.add_node("scrape", scrape_linkedin)
    workflow.add_node("filter", filter_jobs)

    workflow.add_edge(START, "scrape")
    workflow.add_edge("scrape", "filter")
    workflow.add_edge("filter", END)

    graph = workflow.compile()

    for output in graph.stream({"jobs": []}):
        for key, value in output.items():
            pass
