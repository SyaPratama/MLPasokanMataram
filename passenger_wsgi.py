"""
Passenger WSGI entry point for cPanel / LiteSpeed shared hosting deployment.
Bridges ASGI FastAPI application (app.main:app) to standard WSGI interface via a2wsgi.
"""

import os
import sys

# Tambahkan direktori root proyek ke sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from a2wsgi import ASGIMiddleware
from app.main import app as asgi_app

# Phusion Passenger / LiteSpeed di cPanel mengeksekusi objek 'application'
application = ASGIMiddleware(asgi_app)
