from __future__ import annotations

import click
from aiohttp import web

from .app import create_app
from .config import dump_config, load_config


@click.group()
def cli():
    """vLLM multi-instance load balancer"""
    pass


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
def validate(config_path: str | None):
    """Validate balancer config"""
    config = load_config(config_path)
    click.echo("Config valid")
    click.echo(dump_config(config))


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
def serve(config_path: str | None):
    """Start the load balancer server"""
    config = load_config(config_path)
    app = create_app(config_path)
    server = config.get("server", {})
    web.run_app(app, host=server.get("host", "0.0.0.0"), port=server.get("port", 9000))


if __name__ == "__main__":
    cli()
