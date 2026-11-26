"""FastAPI web application and REST API for CI-FaultGuard."""

from cifault.api.app import app, create_app

__all__ = ["app", "create_app"]
