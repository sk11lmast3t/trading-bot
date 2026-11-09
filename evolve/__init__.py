"""Research-only strategy evolution tools.

Importing the package disables python-dotenv's implicit .env loading. LLM
credentials are read only from evolve/.llm_env by the proposal module.
"""
import os

os.environ["PYTHON_DOTENV_DISABLED"] = "1"

