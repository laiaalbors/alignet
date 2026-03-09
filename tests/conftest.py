"""Add alignet/ to sys.path so imports like `from datasets.xxx import ...` resolve."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "alignet"))
