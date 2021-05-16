import torch
import typer
from datasets import load_dataset
from transformers import BertModel, PreTrainedTokenizerFast

from zebraodqa.main import Config

app = typer.Typer()


@app.command()
def ingest():
    """
    Every model from HugginFace is applicable
    TODO: put url here
    Corpus example: squad | MedQA or FindZebra
    """
    typer.secho("Welcome to the ingest command", fg=typer.colors.WHITE, bold=True)

    model = BertModel.from_pretrained(Config['model'].get())
    fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(Config['tokenizer'].get())
    # fast_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    corpus = load_dataset(Config['corpus'].get(),
                          split='train[:100]')  # cache_dir=Config['cache_dir'].get() -- Cache directory override

    torch.set_grad_enabled(False)

    typer.secho("Embedding corpus as dense context vector representation using FAISS.")
    corpus_embeddings = corpus.map(
        lambda example: {
            'embeddings': model(**fast_tokenizer(example['line'], return_tensors='pt'))['pooler_output'][0].numpy()})
    # corpus_embeddings.save_to_disk(os.path.join(Config['cache_dir'].get(), "corpus/"))

    typer.secho("Adding FAISS index for efficient similarity search and clustering of dense vectors.")
    corpus_embeddings.add_faiss_index(column='embeddings')

    typer.secho("Saving the index")
    corpus_embeddings.save_faiss_index("embeddings", "corpus.faiss")  # os.path.join(Config['cache_dir'].get())
    return 0


if __name__ == "__main__":
    app()
