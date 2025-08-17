
from dataclasses import dataclass, field

try:
    from isaaclab.utils import configclass
except ModuleNotFoundError:
    configclass = dataclass