"""Version of the package as determined from environment variable in test/prod."""

import os

__version__ = version = os.environ.get("PACKAGE_VERSION", None) or "0.0.1"
