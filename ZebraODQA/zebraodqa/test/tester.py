# ========================================
#      Similarity test of tokenizers
# ========================================
def testtokenizer():
    vocabfile = "../Bert/assets/vocab.txt"
    test_realm_tokenizer(vocabfile)

# ========================================
#       Similarity test of models
# ========================================
def testreader():
    test_realm_similarity_reader()

    return 0

def testretriever():
    test_realm_similarity_retriever()

    return 0

# ========================================
#           Retriever tests
# ========================================
def testretrieverrealm():
    model = REALMEmbedder.load(BertConfig(intermediate_size=3072), Config.embedder).to(Config.device)

    dataset = Squad().get_validation_data()
    corpus = Squad2(model.tokenizer).get_corpus()

    print("\nEmbedding corpus as dense context vector representations.")
    # corpus_with_embeddings = corpus.map(lambda example: {'embeddings': model(**model.tokenizer(example["context"]["title"], example["context"]["sentence"], return_tensors="pt").to(Config.device))[0][0].cpu().detach().numpy()})

    print("\nAdding Faiss index for efficient similarity search and clustering of dense vectors.")
    # corpus_with_embeddings.add_faiss_index(column="embeddings")
    corpus.load_faiss_index("embeddings", os.path.join(Config.cache_dir, "wikipedia_realm.faiss"))

    test_dense_retriever(model, model.tokenizer, dataset, corpus)

    return 0

def testretrieverdpr():
    dpr_embedder = DPR_embedder()

    dataset = Squad().get_validation_data()
    corpus = Squad2(dpr_embedder.ctx_tokenizer).get_corpus()

    print("\nEmbedding corpus as dense context vector representations.")
    # corpus_with_embeddings = dpr_embedder.embed_context(corpus)
    
    print("\nAdding Faiss index for efficient similarity search and clustering of dense vectors.")
    corpus.load_faiss_index("embeddings", os.path.join(Config.cache_dir, "wikipedia_dpr.faiss"))
    # corpus_with_embeddings.add_faiss_index(column="embeddings")

    test_dense_retriever(dpr_embedder.q_encoder, dpr_embedder.q_tokenizer, dataset, corpus)

    return 0

def testtfidf():
    dpr_embedder = DPR_embedder()
    dataset = Squad().get_validation_data()
    corpus = Squad2(dpr_embedder.ctx_tokenizer).get_corpus()

    corpus.load_elasticsearch_index("ctx", host="localhost", port="9200", es_index_name="wikipediasquad")

    test_sparse_retriever(dataset, corpus)

    return 0

def testembed():
    dpr_embedder = DPR_embedder()

    corpus = Wikipedia(dpr_embedder.ctx_tokenizer).get_corpus()

    print("\nEmbedding corpus as dense context vector representations.")
    dpr_embedder.embed_context(corpus)

    return 0

def testlength():
    dataset = MSMarco().get_train_data()