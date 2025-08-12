"""
Python module serving as a project/extension template.
"""
import os

TRACKERLAB_REPO_DIR = os.path.dirname(os.path.abspath(__file__))

TRACKERLAB_ASSETS_DIR = os.path.join(TRACKERLAB_REPO_DIR, "..", "data", "assets")

TRACKERLAB_USD_DIR = os.path.join(TRACKERLAB_ASSETS_DIR, "usd")

# Register Gym environments.
# from .tasks import *

