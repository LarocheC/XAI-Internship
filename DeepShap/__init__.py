"""DeepShap: attribution tooling for the NsNet2 speech enhancer.

The scripts in this package are run directly (``python DeepShap/main.py``), which puts
``DeepShap/`` itself on sys.path, so the submodules import each other as top-level
modules (``from utils.data_utils import ...``). Re-exporting them here would require the
opposite layout, and the previous ``from ..config.parameters import ...`` raised
ImportError on any attempt to import this package. Kept deliberately empty.
"""
