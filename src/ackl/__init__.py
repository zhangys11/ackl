import importlib.metadata
__version__ = importlib.metadata.version('ackl')

import cla
if not hasattr(cla, '__version__') or cla.__version__ < '2.1.0':
    raise ImportError('ackl requires cla >= 2.1.0. Please upgrade: pip install -U cla')