import os
import logging
import sys
import psycopg2
from sqlalchemy import make_url
from llama_index.core import (
    VectorStoreIndex,
    Document,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.postgres import PGVectorStore

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

DOCUMENT_PATH = "./data/Benefits/BYOD/"
STORAGE_PATH = "./storage/benefit_pipeline_storage_pg"
DB_NAME = "benefit_vector_db"
DB_USER = "benefit_vector_user"
DB_PWD = os.environ.get("DB_PWD")
CONNECTION_STRING = "postgresql://localhost:5432"

if not os.path.exists(DOCUMENT_PATH):
    raise ValueError(f"Document path {DOCUMENT_PATH} does not exist")


def create_db():
    conn = psycopg2.connect(CONNECTION_STRING)
    conn.autocommit = True

    with conn.cursor() as c:
        # c.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
        c.execute(f"CREATE DATABASE {DB_NAME} owner {DB_USER}")


url = make_url(CONNECTION_STRING)


def create_vector_store():
    vector_store = PGVectorStore.from_params(
        database=DB_NAME,
        host=url.host,
        user=DB_USER,
        password=DB_PWD,
        port=url.port,
        table_name="benefits_vector_table",
    )

    return vector_store


def load_documents():
    print("Loading documents...")
    vector_store = create_vector_store()
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
            TitleExtractor(),
            OpenAIEmbedding(),
        ],
        docstore=SimpleDocumentStore(),
        vector_store=vector_store,
    )

    documents = SimpleDirectoryReader(DOCUMENT_PATH).load_data()
    # transform docs using pipeline
    transformed_nodes = pipeline.run(documents=documents)
    pipeline.persist(STORAGE_PATH)
    return vector_store


def build_index(vector_store):
    print("Building index...")
    index = VectorStoreIndex.from_vector_store(vector_store)
    index.storage_context.persist(STORAGE_PATH)
    return index
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # index = VectorStoreIndex.from_documents(
    #     documents, storage_context=storage_context, show_progress=True
    # )
    # index = VectorStoreIndex(
    #     nodes=nodes, storage_context=storage_context, show_progress=True
    # )


def load_index():
    print("Loading index from storage")
    if not os.path.exists(STORAGE_PATH):
        raise ValueError(f"Storage path {STORAGE_PATH} does not exist")

    vector_store = create_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index


def query():
    if not os.path.exists(STORAGE_PATH):
        vector_store = load_documents()
        index = build_index(vector_store)
    else:
        index = load_index()

    query_engine = index.as_query_engine()
    response = query_engine.query("how much is benefit?")
    print(response)


# create_db()

query()

# load the data
# pipeline.load(persist_dir=STORAGE_PATH)
# nodes = pipeline.run()

# index = VectorStoreIndex.from_documents(pipeline.run())
# index = VectorStoreIndex.from_vector_store(
#     vector_store=vector_store,
#     embed_model=embed_model,
# )
