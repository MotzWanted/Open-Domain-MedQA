import typer

app = typer.Typer()

@app.callback()
def callback():
    """
    Awesome CLI Tool!
    """

@app.command()
def main():
    """
    The main screen of the cli tool...
    """
    typer.secho("Welcome to ZebraODQA", fg=typer.colors.WHITE, bold=True)