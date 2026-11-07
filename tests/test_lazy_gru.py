import os
import subprocess
import sys


def test_main_imports_without_tensorflow_when_gru_filter_is_off(tmp_path):
    root = os.path.dirname(os.path.dirname(__file__))
    code = """
import builtins, sys
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'tensorflow' or name.startswith('tensorflow.'):
        raise ImportError('tensorflow intentionally unavailable')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import main
assert 'core.gru_predictor' not in sys.modules
"""
    env = dict(os.environ, PYTHONPATH=root, PYTHON_DOTENV_DISABLED="1")
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True)
