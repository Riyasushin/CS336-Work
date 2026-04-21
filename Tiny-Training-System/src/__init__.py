import importlib.metadata

try:
    __version__ = importlib.metadata.version("tiny-training-system")
except importlib.metadata.PackageNotFoundError:
    pass
