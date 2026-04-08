from torchwright.reference_renderer.types import Segment, RenderConfig
from torchwright.reference_renderer.trig import generate_trig_table
from torchwright.reference_renderer.render import render_frame, save_png
from torchwright.reference_renderer.scenes import box_room, multi_room

__all__ = [
    "render_frame",
    "save_png",
    "Segment",
    "RenderConfig",
    "generate_trig_table",
    "box_room",
    "multi_room",
]
