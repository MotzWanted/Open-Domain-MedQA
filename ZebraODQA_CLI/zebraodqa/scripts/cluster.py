import click

from realm.realm_embedder import REALMEmbedder


@click.command()
def home():
    click.echo("Hello and welcome to cluster control!")

@click.command()
def prepmedqarealm():
    """Prepares MedQA with the REALM embedder"""
    realm = REALMEmbedder.load(BertConfig(intermediate_size=3072), config_embedder).to(device)

    corpus = MedQACorpus(realm.tokenizer).get_corpus()

    print("\nEmbedding corpus as dense context vector representations.")
    corpus_with_embeddings = realm.embed_context(corpus)
    corpus_with_embeddings.save_to_disk(os.path.join(cache_dir, "realmmedqa/"))
    
    print("\nAdding Faiss index for efficient similarity search and clustering of dense vectors.")
    corpus_with_embeddings.add_faiss_index(column="embeddings")

    # Save index
    print(f"\nSaving the index to {os.path.join(cache_dir, 'medqa_realm.faiss')}")
    corpus_with_embeddings.save_faiss_index("embeddings", os.path.join(cache_dir, "medqa_realm.faiss"))

    return 0

@click.command()
def prepfindzebrarealm():
    """Prepares FindZebra with the REALM embedder"""
    realm = REALMEmbedder.load(BertConfig(intermediate_size=3072), config_embedder).to(device)

    # replace with FindZebra
    corpus = Squad2(realm.tokenizer).get_corpus()

    print("\nEmbedding corpus as dense context vector representations.")
    corpus_with_embeddings = realm.embed_context(corpus)
    corpus_with_embeddings.save_to_disk(os.path.join(cache_dir, "realmmfindzebra/"))
    
    print("\nAdding Faiss index for efficient similarity search and clustering of dense vectors.")
    corpus_with_embeddings.add_faiss_index(column="embeddings")

    # Save index
    print(f"\nSaving the index to {os.path.join(cache_dir, 'findzebra_realm.faiss')}")
    corpus_with_embeddings.save_faiss_index("embeddings", os.path.join(cache_dir, "findzebra_realm.faiss"))

    return 0

@click.command()
def prepdatasets():
    # Squad().get_train_data()
    # NQ().get_train_data()
    # TriviaQA().get_train_data()

    MedQA().get_train_data()

    return 0


@click.command()
def train():
    dpr_reader = DPR_reader()
    dpr_embedder = DPR_embedder()

    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(Config.device)

    trainer = QATrainer(dpr_reader.r_encoder,
        dpr_embedder.q_encoder,
        dpr_reader.r_tokenizer,
        dpr_embedder.q_tokenizer,
        MedQA(dpr_reader.r_tokenizer).get_train_data(),
        MedQA(dpr_reader.r_tokenizer).get_validation_data(),
        MedQACorpus(dpr_embedder.ctx_tokenizer).get_corpus()
        #Squad(dpr_reader.r_tokenizer).get_train_data(),
        #Squad(dpr_reader.r_tokenizer).get_validation_data(),
        #Wikipedia(dpr_embedder.ctx_tokenizer).get_corpus()
    )

    reader, retriever = trainer.train()

    torch.save(reader.state_dict(), os.path.join(config_cache_dir, "trainedreader.pth.tar"))
    torch.save(retriever.state_dict(), os.path.join(config_cache_dir, "trainedretriever.pth.tar"))

    return 0