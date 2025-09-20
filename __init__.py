# This file makes the folder a Python package
# It re-exports NODE_CLASS_MAPPINGS so ComfyUI can see them.

from .model_merge_combos import NODE_CLASS_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS']
