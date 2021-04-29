import typer

app = typer.Typer()

@app.command()
def test():
    """
    A test screen of the cli tool...
    """
    typer.secho("This is a test!", fg=typer.colors.RED, bold=True)