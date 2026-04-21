import importlib.metadata
import pathlib

__version__ = importlib.metadata.version("cs336_basics")
HOME = pathlib.Path.home()
STANFORD_CS_ROOT = HOME.joinpath("software/sound_thinking/stanford_cs")
CS336_BASICS_ROOT = STANFORD_CS_ROOT.joinpath("cs336-assignment1-basics")
