import torch
import typer
from transformers import BertModel, TrainingArguments, Trainer

from model.qatrainer import QATrainer
from zebraodqa.main import Config

app = typer.Typer()
@app.command()
def finetune():
    """
    FineTune the model for QuestionAnswering
    """

    model = BertModel.from_pretrained('bert-base-uncased')
    batch_size = Config['batch_size'].get()


    test:Trainer = QATrainer(True)

    training_args = TrainingArguments()

    return 0

if __name__ == "__main__":
    app()
