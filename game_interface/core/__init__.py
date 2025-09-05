"""
Core Game Interface Package

Contains the core game interface classes and parsers:
- ZorkInterface: Direct interface to dfrotz processes
- StructuredZorkParser: Parser for Zork's structured output format
"""

from .zork_interface import ZorkInterface
from .structured_parser import StructuredZorkParser, StructuredZorkResponse

__all__ = ["ZorkInterface", "StructuredZorkParser", "StructuredZorkResponse"]
