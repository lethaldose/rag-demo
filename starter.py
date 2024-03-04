import os

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

print(Settings.chunk_size)

# storage_path = "./storage"
# if os.path.exists(storage_path):
#     # read storage context from existing index
#     storage_context = StorageContext.from_defaults(persist_dir=storage_path)
#     index = load_index_from_storage(storage_context)
# else:
#     # create a new index
# documents = SimpleDirectoryReader("data").load_data()
# index = VectorStoreIndex.from_documents(documents)
# index.storage_context.persist(storage_path)

# query_engine = index.as_query_engine()
# response = query_engine.query("what did the author do while growing up?")
# print(response)
