from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    Settings,
)
from llama_index.readers.smart_pdf_loader import SmartPDFLoader
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import psycopg2
import os
from sqlalchemy import make_url


def load_index(reload: bool) -> VectorStoreIndex:
    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    connection_string = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5432"
    )
    db_name = "job_agent"
    url = make_url(connection_string)
    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=url.host,
        password=url.password,
        port=url.port,
        user=url.username,
        table_name="candidate",
        embed_dim=768,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )

    # Embedding model
    ollama_embedding = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0},
    )
    Settings.embed_model = ollama_embedding

    if reload:
        # Load documents
        # documents = SimpleDirectoryReader('./data').load_data()
        llmsherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all"
        pdf_url = "data/ZuoyunZhengCVplain.pdf"
        documents = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url).load_data(
            pdf_url
        )
        import pdb

        pdb.set_trace()

        # Persistent Storage
        conn = psycopg2.connect(connection_string)
        conn.autocommit = True
        with conn.cursor() as c:
            c.execute(f"DROP DATABASE IF EXISTS {db_name}")
            c.execute(f"CREATE DATABASE {db_name}")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Build index
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, show_progress=True
        )
        index.storage_context.persist()
    else:
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir="./storage"
        )
        index = load_index_from_storage(storage_context=storage_context)

    return index


if __name__ == "__main__":
    from dotenv import load_dotenv
    from utils.args import parse_args

    load_dotenv()
    args = parse_args()

    Settings.llm = Ollama(
        model="deepseek-r1:1.5b",
        request_timeout=30,
    )
    index = load_index(args.reload_data)
    query_engine = index.as_query_engine()
    print(query_engine.query("What is the candidate's name?"))
    print(query_engine.query("What is the candidate's contact information?"))
    print(
        query_engine.query(
            "What is the candidate's qualification and skills for a Machine Learning Engineer position?"
        )
    )
