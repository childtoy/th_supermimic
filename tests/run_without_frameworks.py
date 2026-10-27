"""Run the real CLI while actively rejecting external environment frameworks."""
import importlib.abc
from pathlib import Path
import runpy
import sys

class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'unilab','unisim','mjlab','isaaclab'}:
            raise RuntimeError(f'Unexpected environment framework import: {fullname}')

sys.meta_path.insert(0,BlockFrameworks())
runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/train.py'),run_name='__main__')
