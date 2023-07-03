import importlib.metadata
__version__ = importlib.metadata.version('ackl')

import cla
assert( '__version__' in cla.__dict__ and cla.__version__ >= '1.3.0')