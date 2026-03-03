"""Version of the package as determined from environment variable in test/prod."""

import os

version = __version__ = os.getenv("PACKAGE_PROD_VERSION", None) or "0.0.1"
