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
from commands.ingest import ingest
from alive_progress import alive_bar

app = typer.Typer()
app.add_typer(ingest.app, name="ingest")

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