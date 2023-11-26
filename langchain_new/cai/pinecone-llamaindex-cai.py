# Build city document index
from llama_index.storage.storage_context import StorageContext


city_indices = {}
for pinecone_title, wiki_title in zip(pinecone_titles, wiki_titles):
    metadata_filters = {"wiki_title": wiki_title}
    vector_store = PineconeVectorStore(
        index_name=index_name,
        environment=environment,
        metadata_filters=metadata_filters,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    city_indices[wiki_title] = VectorStoreIndex.from_documents(
        city_docs[wiki_title],
        storage_context=storage_context,
        service_context=service_context,
    )
    # set summary text for city
    city_indices[wiki_title].index_struct.index_id = pinecone_title
