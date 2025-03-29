from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from jobspy_tools import jobspy_scrape_jobs
from utils.args import parse_args


load_dotenv()


def main(args):
    # Settings
    llm = Ollama(
        model="deepseek-r1:1.5b",
        request_timeout=120,
    )
    Settings.llm = llm
    # Index for RAG
    # index = load_index(args.reload_data)

    # Search
    jobspy_tool = FunctionTool.from_defaults(
        jobspy_scrape_jobs,
        name="jobspy",
        description="Scrape jobs from job boards such as LinkedIn and glassdoor",
    )

    agent = ReActAgent.from_tools(llm=llm, tools=[jobspy_tool])
    response = agent.chat(
        "Can you find 2 jobs for me with the title AI Engineer in Germany from LinkedIn within the past week?"
    )
    print(response)


if __name__ == "__main__":
    main(parse_args())
