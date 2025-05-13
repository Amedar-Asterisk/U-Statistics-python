"""
Tensor Contraction Package

This package provides tools for efficient tensor network contractions and computations.
It includes classes for managing tensor contraction states and performing contractions
with different backends (numpy/torch).

Main Components:
    - TensorContractionCalculator: Handles tensor contractions with multiple backend options
    - TensorContractionState: Manages the state and optimization of tensor contractions

Example:
    >>> from tensor_contraction import TensorContractionCalculator
    >>> calculator = TensorContractionCalculator()
    >>> result = calculator.contract_tensors(tensors, computing_sequence)
"""

from .calculator import TensorContractionCalculator

__version__ = "0.1.0"
__author__ = "Zhang RuiQi"

__all__ = [
    "TensorContractionCalculator",
    "TensorContractionState",
]
