"""Deterministic UDFs for a Video -> Frame -> Caption -> Video workload."""

from __future__ import annotations


class DecodeFrames:
    def run(self, videos):
        return [
            [{"video": name, "frame": frame} for frame in range(frame_count)]
            for name, frame_count in videos
        ]


class CaptionFrames:
    def run(self, frames):
        return [
            {
                **frame,
                "caption": f"{frame['video']}:caption:{frame['frame']}",
            }
            for frame in frames
        ]


class SummarizeCaptions:
    def run(self, caption_groups):
        return [
            {
                "video": captions[0]["video"],
                "frames": tuple(caption["frame"] for caption in captions),
                "captions": tuple(caption["caption"] for caption in captions),
            }
            for captions in caption_groups
        ]


__all__ = ["CaptionFrames", "DecodeFrames", "SummarizeCaptions"]
