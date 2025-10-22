# src/hsi_model/models/generator.py
"""
Generator wrapper exposing the production CSWin generator implementation.
"""

from .generator_v3 import NoiseRobustCSWinGenerator

__all__ = ["NoiseRobustCSWinGenerator"]
