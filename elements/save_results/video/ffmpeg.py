import subprocess
from typing import List, Optional

import numpy as np

from elements.save_results.video.base import VideoWriterBase


class FFmpegWriter(VideoWriterBase):
    """
    FFmpeg-based video writer that encodes raw RGB frames via a subprocess
    and writes them to a video file. Frames are passed via stdin as raw bytes.

    Useful when:
        - Full control over codec, CRF, presets, pixel format
        - GPU acceleration (h264_nvenc, hevc_nvenc)
        - OpenCV VideoWriter is insufficient or undesirable

    :param path: Output file path, e.g., "/path/to/output.mp4"
    :param w: Frame width in pixels
    :param h: Frame height in pixels
    :param fps: Target frames per second
    :param codec: Video codec to use. Examples: "libx264", "h264_nvenc", "hevc_nvenc", "libx265"
    :param pix_fmt: Pixel format for the output video. Examples: "yuv420p", "yuv444p", "nv12"
    :param crf: Constant Rate Factor for quality control (0–51). Lower = higher quality. Ignored by some hardware encoders
    :param bitrate: Optional target bitrate (e.g., "8M"). Overrides CRF if set
    :param preset: Encoder speed/quality tradeoff. Examples: "slow", "medium", "fast" (CPU), "p1"–"p7" (NVENC)
    """
    def __init__(
        self,
        path: str,
        w: int,
        h: int,
        fps: float,
        codec: str = "hevc_nvenc",
        pix_fmt: str = "yuv420p",
        crf: int = 23,
        bitrate: Optional[str] = None,
        preset: str = "fast",
    ):
        self.path = path
        self.w = w
        self.h = h
        self.codec = codec
        self.pix_fmt = pix_fmt
        self.fps = fps

        # ------------------------------------------------------------------
        # Build command
        # ------------------------------------------------------------------
        cmd: List[str] = [
            "ffmpeg",
            "-y",  # overwrite output
            "-f",
            "rawvideo",  # raw RGB frames coming from stdin
            "-pix_fmt",
            "rgb24",  # input format (raw frame format)
            "-s",
            f"{w}x{h}",  # width x height
            "-r",
            str(fps),  # input frame rate
            "-i",
            "pipe:0",  # read raw frames from stdin

            # Output settings
            "-vcodec",
            codec,
            "-pix_fmt",
            pix_fmt,
        ]

        # Compression settings
        if "nvenc" not in codec:
            # CPU encoders (libx264/libx265) use CRF
            cmd += ["-crf", str(crf)]
        if bitrate is not None:
            # Bitrate override (ignored by some encoders if CRF is set)
            cmd += ["-b:v", bitrate]

        # Encoder preset (varies by codec)
        cmd += ["-preset", preset]

        # Output path
        cmd += [path]

        # ------------------------------------------------------------------
        # Spawn ffmpeg process
        # ------------------------------------------------------------------
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=10**8,
        )

    # ------------------------------------------------------------------
    def write(self, frame: np.ndarray) -> None:
        """
        Write a single RGB frame to FFmpeg via stdin.

        :param frame: RGB uint8 array of shape (h, w, 3)
        :raises RuntimeError: If FFmpeg pipe is broken
        :raises ValueError: If frame shape or dtype is invalid
        :returns: None
        """
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            raise ValueError(f"Frame size mismatch: got {frame.shape[1]}x{frame.shape[0]} "
                             f"expected {self.w}x{self.h}")
        if frame.dtype != np.uint8:
            raise ValueError("FFmpegWriter expects uint8 RGB frames")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frame must have shape (H, W, 3)")

        try:
            self.proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            err = self.proc.stderr.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"FFmpeg pipe broken. stderr:\n{err}")

    # ------------------------------------------------------------------
    def close(self) -> None:
        """
        Flush the FFmpeg pipeline and terminate the process.

        :raises RuntimeError: If FFmpeg exits with a non-zero code
        :returns: None
        """
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except Exception:
                pass

        self.proc.wait(timeout=5)
        if self.proc.returncode != 0:
            err = self.proc.stderr.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"FFmpeg exited with code {self.proc.returncode}:\n{err}")
