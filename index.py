from langchain_postgres.vectorstores import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph
import psycopg2
import os
from dotenv import load_dotenv
from utils.args import parse_args


def load_index(reload: bool):
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

    # LLM model
    llm = OllamaLLM(
        model="deepseek-r1:1.5b",
        # request_timeout=30,
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
        conn = psycopg2.connect(connection_string)
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

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    # Create a basic LangGraph for the retrieval QA system
    prompt_template = """
    Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Define retrieval chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Create a simple LangGraph
    workflow = StateGraph("rag_workflow")

    # Define the nodes
    def retrieval_node(state):
        question = state["question"]
        answer = rag_chain.invoke(question)
        return {"answer": answer}

    # Add nodes
    workflow.add_node("retrieval", retrieval_node)

    # Add edges
    workflow.set_entry_point("retrieval")
    workflow.set_finish_point("retrieval")

    # Compile the graph
    graph = workflow.compile()

    return graph, retriever


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()

    graph, retriever = load_index(args.reload_data)

    # Test queries
    questions = [
        "What is the candidate's name?",
        "What is the candidate's contact information?",
        "What is the candidate's qualification and skills for a Machine Learning Engineer position?",
    ]

    for question in questions:
        result = graph.invoke({"question": question})
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")
        print("-" * 50)
