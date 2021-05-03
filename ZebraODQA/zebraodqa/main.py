import os

import transformers
import typer
import confuse
import yaml
from tokenizers import Tokenizer
from transformers import BertModel, PreTrainedTokenizerFast, BertTokenizer, DPRContextEncoder, \
    DPRContextEncoderTokenizer, pipeline, DPRQuestionEncoder, BertTokenizerFast, AutoTokenizer
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
    return 0

## Finetuning section
@app.command()
def finetune(freezeLayers: bool, config: bool):
    """
    Training command command!
    """
    model = BertModel.from_pretrained('bert-base-uncased' if not config else Config['model'].get())
    if torch.cuda.is_available():
        model = model.to(torch.device("device"))


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
    #fast_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    corpus = load_dataset(Config['corpus'].get(), split='train[:100]') # cache_dir=Config['cache_dir'].get() -- Cache directory override

    torch.set_grad_enabled(False)

    typer.secho("Embedding corpus as dense context vector representation using FAISS.")
    corpus_embeddings = corpus.map(
        lambda example: {'embeddings': model(**fast_tokenizer(example['line'], return_tensors='pt'))['pooler_output'][0].numpy()})
    # corpus_embeddings.save_to_disk(os.path.join(Config['cache_dir'].get(), "corpus/"))

    typer.secho("Adding FAISS index for efficient similarity search and clustering of dense vectors.")
    corpus_embeddings.add_faiss_index(column='embeddings')

    typer.secho("Saving the index")
    # corpus_embeddings.save_faiss_index("embeddings", os.path.join(Config['cache_dir'].get(), "corpus.faiss"))
    return 0


@app.command()
def config_template():
    """
    Dump the config template file for edit
    """
    Config.dump('config.yaml', redact=True)

@app.callback()
def main(config: typer.FileText = typer.Option(None) , verbose: bool = False):
    """
    The main screen of the zebraodqa tool...
    """
    if verbose:
        typer.secho("Verbose output turned on", fg=typer.colors.RED)
        state["verbose"] = True

    if config != None:
        Config.set_file(config)
        typer.secho("Successfully loaded config file", fg=typer.colors.GREEN)