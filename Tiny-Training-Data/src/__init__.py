import importlib.metadata

try:
    __version__ = importlib.metadata.version("tiny-training-data")
except importlib.metadata.PackageNotFoundError:
    pass
