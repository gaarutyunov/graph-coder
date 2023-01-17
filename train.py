import click
from catalyst.contrib.scripts.run import run_from_config


@click.command()
@click.option("--config", default="configs/small.yaml", help="Path to config file")
def main(config: str):
    run_from_config(configs=[config])


if __name__ == "__main__":
    main()
