"""A Python package for robotic arm control and automation."""

from __future__ import annotations

__version__ = "0.1.0"

from . import damiao

try:
    from . import simulation
except ImportError:
    simulation = None  # mujoco/glfw not available

try:
    from . import netcan
except ImportError:
    netcan = None  # python-can not installed

__all__ = ["damiao", "netcan", "simulation"]
