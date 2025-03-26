from llama_index.core import SimpleDirectoryReader, StorageContext 
from llama_index.core import VectorStoreIndex, Settings
from llama_index.readers.smart_pdf_loader import SmartPDFLoader
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import textwrap
import psycopg2
from dotenv import load_dotenv
import os
from sqlalchemy import make_url

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# Load documents
#documents = SimpleDirectoryReader('./data').load_data()
llmsherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all"
pdf_url = "data/ZuoyunZhengCV.pdf"
documents = SmartPDFLoader(llmsherpa_api_url=llmsherpa_api_url).load_data(pdf_url)

# Involatile Storage
connection_string = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5432"
db_name = "job_agent"
conn = psycopg2.connect(connection_string)
conn.autocommit = True

with conn.cursor() as c:
    c.execute(f"DROP DATABASE IF EXISTS {db_name}")
    c.execute(f"CREATE DATABASE {db_name}")

url = make_url(connection_string)
vector_store = PGVectorStore.from_params(
    database=db_name,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="paul_graham_essay",
    embed_dim=768,  # openai embedding dimension
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Embedding model
ollama_embedding = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)
Settings.embed_model = ollama_embedding

# Build index
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, 
    show_progress=True
)

# LLM
Settings.llm = Ollama(
    model="deepseek-r1:1.5b",
    request_timeout=30,
)
query_engine = index.as_query_engine()
import pdb; pdb.set_trace()
print(query_engine.query("What is the candidate's name?"))
print(query_engine.query("What is the candidate's contact information?"))

