import click

@click.group()
def configuration():
    click.echo("Welcome to the configuration section")

@click.command()
def conf_model():
    click.echo("Configure system")

@click.command()
def conf_dataset():
    click.echo("Configure dataset")

@click.command()
def home():
    click.echo("Welcome to home")
