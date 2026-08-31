"""Executable entry point."""

import logging

import uvicorn

from qwen_a2a.app import create_app
from qwen_a2a.config import get_settings


def create_app_from_env():
    return create_app(get_settings())


def run() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        "qwen_a2a.main:create_app_from_env",
        factory=True,
        host=settings.a2a_host,
        port=settings.a2a_port,
    )


if __name__ == "__main__":
    run()
