import os

import transformers
import typer
import confuse
from transformers import BertModel, BertTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
from datasets import load_dataset
import torch
from alive_progress import alive_bar

app = typer.Typer()
state = {"verbose": False}
Config = confuse.Configuration('ZebraODQA', __name__)

## Bert section
@app.command()
def bert():
    typer.secho("Bert commands!")


## Corpus section
@app.command()
def corpus(model: str):
    """
    Main corpus command!
    """


@app.command()
def ingest():
    """
    Every model from HugginFace is applicable
    TODO: put url here
    Corpus example: squad | MedQA or FindZebra
    """
    typer.secho("Welcome to the ingest command", fg=typer.colors.WHITE, bold=True)

    #model = BertModel.from_pretrained(pretrainedModel)
    #tokenizer = BertTokenizer.from_pretrained(pretrainedModel)

    torch.set_grad_enabled(False)

    with alive_bar(4) as bar:
        encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base") # optional
        tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base") # optional
        corpus = load_dataset('crime_and_punish', split='train[:100]', cache_dir=Config['cache_dir'].get())  # optional
        bar()

        typer.secho("Embedding corpus as dense context vector representation using FAISS.")
        corpus_embeddings = corpus.map(lambda example: {'embeddings': encoder(**tokenizer(example['line'], return_tensors='pt'))[0][0].numpy()})
        #corpus_embeddings.save_to_disk(os.path.join(Config['cache_dir'].get(), "corpus/"))
        bar()

        typer.secho("Adding FAISS index for efficient similarity search and clustering of dense vectors.")
        corpus_embeddings.add_faiss_index(column='embeddings')
        bar()

        typer.secho("Saving the index")
        #corpus_embeddings.save_faiss_index("embeddings", os.path.join(Config['cache_dir'].get(), "corpus.faiss"))
        bar()
        return 0


@app.callback()
def main(verbose: bool = False):
    """
    The main screen of the zebraodqa tool...
    """
    if verbose:
        typer.secho("Verbose output turned on")
        state["verbose"] = True
