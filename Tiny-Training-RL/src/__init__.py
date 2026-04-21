import importlib.metadata

try:
    __version__ = importlib.metadata.version("tiny-training-rl")
except importlib.metadata.PackageNotFoundError:
    pass
