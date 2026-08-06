"""Real video-workload adapters for the Multigrain v3.5 runtime."""

from .caption_v35 import VideoCaptionV35Pipeline, run_caption_v35
from .multimodal_v35 import VideoMultimodalV35Pipeline, run_multimodal_v35
from .v35 import VideoV35Pipeline, run_v35

__all__ = [
    "VideoCaptionV35Pipeline",
    "VideoMultimodalV35Pipeline",
    "VideoV35Pipeline",
    "run_caption_v35",
    "run_multimodal_v35",
    "run_v35",
]
