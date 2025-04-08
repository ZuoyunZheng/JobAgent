import os

import psycopg
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector


def load_retriever(reload: bool):
    """Load or create vector store index using LangChain."""
    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    connection_string = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5432"
    )
    db_name = "job_agent_langchain"

    # Embedding model
    embed_model = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
        # ollama_additional_kwargs={"mirostat": 0}
    )

    connection_details = {
        "driver": "psycopg",
        "host": "localhost",
        "port": "5432",
        "database": db_name,
        "user": POSTGRES_USER,
        "password": POSTGRES_PASSWORD,
    }

    if reload:
        # Create database if it doesn't exist
        conn = psycopg.connect(connection_string)
        conn.autocommit = True
        with conn.cursor() as c:
            c.execute(f"DROP DATABASE IF EXISTS {db_name}")
            c.execute(f"CREATE DATABASE {db_name}")
            c.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.close()

        # Load documents
        pdf_path = "data/ZuoyunZhengCVplain.pdf"
        raw_documents = PyPDFLoader(pdf_path).load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
        documents = text_splitter.split_documents(raw_documents)

        # Create vector store
        vector_store = PGVector.from_documents(
            documents=documents,
            embedding=embed_model,
            collection_name="candidate",
            pre_delete_collection=True,
            distance_strategy="cosine",
            connection=PGVector.connection_string_from_db_params(
                **connection_details,
            ),
        )
    else:
        # Load existing vector store
        vector_store = PGVector(
            collection_name="candidate",
            embeddings=embed_model,
            connection=PGVector.connection_string_from_db_params(
                **connection_details,
            ),
        )

    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


if __name__ == "__main__":
    from typing import Annotated, Sequence

    from dotenv import load_dotenv
    from langchain import hub
    from langchain_core.messages import BaseMessage, HumanMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_ollama import ChatOllama
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from typing_extensions import TypedDict

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
    class AgentState(TypedDict):
        # The add_messages function defines how an update should be processed
        # Default is to replace. add_messages says "append"
        messages: Annotated[Sequence[BaseMessage], add_messages]

    # Nodes
    def retrieve(state):
        print("----RETRIEVE----")
        messages = state["messages"]
        question = messages[0].content
        # response = retrieval_llm.invoke(question)
        docs = retriever.invoke(question)

        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        return {"messages": [format_docs(docs)]}

    def generate(state):
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]
        docs = last_message.content

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")

        # Chain
        rag_chain = prompt | llm | StrOutputParser()

        # Run
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}

    # Graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    # Add edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    graph = workflow.compile()

    # Test queries
    questions = [
        "What is the applicant's name?",
        "What is the applicant's contact information and address?",
        "Where is the applicant based?",
        "What is the candidate's qualifications and skills for a Machine Learning Engineer position?",
    ]

    for question in questions:
        result = graph.invoke({"messages": [HumanMessage(content=question)]})
        print(f"Question: {question}")
        print(f"Answer: {result['messages'][-1].content}")
        print("-" * 50)
