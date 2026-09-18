"""Deterministic UDFs for sibling audio and frame relations."""

from __future__ import annotations


class DecodeAudio:
    def run(self, videos):
        return [
            [{"video": name, "chunk": chunk} for chunk in range(audio_chunks)]
            for name, _frame_count, audio_chunks in videos
        ]


class TranscribeAudio:
    def run(self, chunks):
        return [
            {
                **chunk,
                "text": f"{chunk['video']}:audio:{chunk['chunk']}",
            }
            for chunk in chunks
        ]


class SummarizeAudio:
    def run(self, transcript_groups):
        return [
            {
                "video": transcripts[0]["video"],
                "audio": tuple(item["chunk"] for item in transcripts),
            }
            for transcripts in transcript_groups
        ]


class DecodeFrames:
    def run(self, videos):
        return [
            [{"video": name, "frame": frame} for frame in range(frame_count)]
            for name, frame_count, _audio_chunks in videos
        ]


class ProcessFrames:
    def run(self, frames):
        return [
            {
                **frame,
                "feature": f"{frame['video']}:vision:{frame['frame']}",
            }
            for frame in frames
        ]


class SummarizeFrames:
    def run(self, feature_groups):
        return [
            {
                "video": features[0]["video"],
                "vision": tuple(item["frame"] for item in features),
            }
            for features in feature_groups
        ]


class MergeModalities:
    def run(self, audio, vision):
        return [
            {
                "video": audio_item["video"],
                "audio": audio_item["audio"],
                "vision": vision_item["vision"],
            }
            for audio_item, vision_item in zip(audio, vision, strict=True)
        ]


__all__ = [
    "DecodeAudio",
    "DecodeFrames",
    "MergeModalities",
    "ProcessFrames",
    "SummarizeAudio",
    "SummarizeFrames",
    "TranscribeAudio",
]
