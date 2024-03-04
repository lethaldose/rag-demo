import os
import logging
import sys
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
from llama_index.core.vector_stores import SimpleVectorStore

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

DOCUMENT_PATH = "./data/Benefits/BYOD/"
STORAGE_PATH = "./storage/benefit_pipeline_storage"

if not os.path.exists(DOCUMENT_PATH):
    raise ValueError(f"Document path {DOCUMENT_PATH} does not exist")


def load_documents():
    print("Loading documents...")
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
            TitleExtractor(),
            OpenAIEmbedding(),
        ],
    )
    documents = SimpleDirectoryReader(DOCUMENT_PATH).load_data()
    # transform docs using pipeline
    transformed_nodes = pipeline.run(documents=documents)
    pipeline.persist(STORAGE_PATH)
    return transformed_nodes


def build_index(nodes):
    print("Building index...")
    index = VectorStoreIndex(nodes=nodes)
    index.storage_context.persist(STORAGE_PATH)
    return index


def load_index():
    print("Loading index from storage")

    if not os.path.exists(STORAGE_PATH):
        raise ValueError(f"Storage path {STORAGE_PATH} does not exist")
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_PATH)
    index = load_index_from_storage(storage_context)
    return index


if not os.path.exists(STORAGE_PATH):
    transformed_nodes = load_documents()
    index = build_index(transformed_nodes)
else:
    index = load_index()

query_engine = index.as_query_engine()
response = query_engine.query("how much is benefit?")
print(response)


# load the data
# pipeline.load(persist_dir=STORAGE_PATH)
# nodes = pipeline.run()

# index = VectorStoreIndex.from_documents(pipeline.run())
# index = VectorStoreIndex.from_vector_store(
#     vector_store=vector_store,
#     embed_model=embed_model,
# )
