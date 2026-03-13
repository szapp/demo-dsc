"""Version of the package as determined from environment variable in test/prod."""

import os

SERVICE = "project-name"
PACKAGE = (__package__ or __name__).split(".", 1)[0]
VERSION = os.environ.get("PACKAGE_PROD_VERSION") or "0.dev0"
