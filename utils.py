from typing import Generic, TypeVar

import taichi as ti
from taichi.lang.struct import StructType

T = TypeVar('T')

# make IntelliSense happy
class TypedStructType(Generic[T], StructType): ...

def ti_dataclass(cls: type[T]) -> TypedStructType[T]:
    return ti.dataclass(cls)

def typing(_: TypedStructType[T], obj) -> type[T]:
    return obj

def transpose(rows: list[tuple]):
    return [list(col) for col in zip(*rows)]