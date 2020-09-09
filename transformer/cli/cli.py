import click

from transformer.train import train as _train


@click.group()
def main_group(**kwargs):
    pass


@main_group.command()
@click.option('--dataset',
              help='specify the dataset to use',
              default='pg-19',
              show_default=True)
def train(**kwargs):
    _train(**kwargs)
