"""Real video-workload adapters for the Multigrain v3.6 runtime."""

from .caption_v36 import VideoCaptionV36Pipeline, run_caption_v36
from .multimodal_v36 import VideoMultimodalV36Pipeline, run_multimodal_v36
from .v36 import VideoV36Pipeline, run_v36

__all__ = [
    "VideoCaptionV36Pipeline",
    "VideoMultimodalV36Pipeline",
    "VideoV36Pipeline",
    "run_caption_v36",
    "run_multimodal_v36",
    "run_v36",
]
