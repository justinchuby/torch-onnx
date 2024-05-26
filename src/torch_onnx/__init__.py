__all__ = ["exported_program_to_ir", "patch_torch"]

from ._patch import patch_torch
from ._core import exported_program_to_ir
