"""
Entrypoint for running core AI as a module with `python -m core`.
"""

if __name__ == "__main__":
    from .config.settings import get_settings
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "core.api.main:app",
        host=settings.host,
        port=settings.port,
    )
