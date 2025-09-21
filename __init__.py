# This file makes the folder a Python package
# It re-exports NODE_CLASS_MAPPINGS so ComfyUI can see them.

from .model_merge_combos import NODE_CLASS_MAPPINGS as MODEL_COMBO_NODES
from .lfgg_ksampler_config import NODE_CLASS_MAPPINGS as LFGG_KSAMPLER_NODES
from .lfgg_resolution_tools import NODE_CLASS_MAPPINGS as LFGG_RESOLUTION_NODES

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(MODEL_COMBO_NODES)
NODE_CLASS_MAPPINGS.update(LFGG_KSAMPLER_NODES)
NODE_CLASS_MAPPINGS.update(LFGG_RESOLUTION_NODES)

__all__ = ['NODE_CLASS_MAPPINGS']