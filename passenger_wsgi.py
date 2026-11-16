"""Passenger WSGI entry point for the FastAPI application."""

import os
import sys
import threading

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from a2wsgi import ASGIMiddleware
from app.main import app as asgi_app

_adapter = None
_adapter_lock = threading.Lock()


def application(environ, start_response):
    global _adapter
    if _adapter is None:
        with _adapter_lock:
            if _adapter is None:
                # Passenger imports before forking; start a2wsgi's loop in the worker.
                _adapter = ASGIMiddleware(asgi_app)
    return _adapter(environ, start_response)
