import importlib.metadata

try:
    __version__ = importlib.metadata.version("tiny-training-basic")
except importlib.metadata.PackageNotFoundError:
    pass
