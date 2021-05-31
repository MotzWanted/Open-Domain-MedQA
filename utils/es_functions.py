import sys
from elasticsearch import Elasticsearch

es = Elasticsearch(timeout=30) # ElasticSearch instance

def es_create_index(index_name:str):
    """
    Create ElasticSearch Index
    """
    es.indices.create(index=index_name)

def es_remove_index(index_name:str):
    """
    Remove ElasticSearch Index
    """
    es.indices.delete(index=index_name)

def es_ingest(index_name:str, title:str, paragraph:str):
    """
    Ingest to ElasticSearch Index
    """
    doc = {
        'title': title,
        'text': paragraph
    }
    response = es.index(index=index_name, body=doc)
    return response

def es_search(index_name:str, query:str, results:int):
    """
    Search in ElasticSearch Index
    """
    response = es.search(
        index=index_name,
        body={
            "query": {"match": {"text": query.lower()}},
            "from": 0,
            "size": results
        })
    
    return response['hits'] # (object) Contains returned documents and metadata.