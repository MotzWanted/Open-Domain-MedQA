import torch
import typer
from transformers import BertModel

app = typer.Typer()
@app.command()
def finetune():
    """
    FineTune the model
    """
    model = BertModel.from_pretrained('bert-base-uncased' if not config else Config['model'].get())
    if torch.cuda.is_available():
        model = model.to(torch.device("device"))
    return 0

if __name__ == "__main__":
    app()
