"""Tyche — a configurable JAX-compatible PRNG using tensor matrix multiplication."""

from tyche.interface import tyche_prng_impl as impl
from tyche.config import TycheConfig

__all__ = ["impl", "TycheConfig"]