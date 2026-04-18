"""Headless pipeline session — runs FrameProcessor without WebRTC.

Headless sessions own a FrameProcessor that drives output sinks (NDI, Spout)
directly. To allow WebRTC viewers to preview the same output without
duplicating the pipeline, the session also exposes its FrameProcessor through
a `VideoProcessingTrack` + `MediaRelay`. Viewers attach via
`POST /api/v1/viewer/attach`, which `relay.subscribe()`s to the same source.

The internal `_consume_frames` task reads from its own relay subscription so
the underlying track's `recv()` loop is always being driven (otherwise frames
would only flow when a viewer was attached).
"""

import asyncio
import logging
import threading
from typing import TYPE_CHECKING

from aiortc.contrib.media import MediaRelay

from .tracks import VideoProcessingTrack

if TYPE_CHECKING:
    from .frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


class HeadlessSession:
    """Pipeline session without WebRTC. Runs FrameProcessor directly."""

    def __init__(
        self,
        frame_processor: "FrameProcessor",
    ):
        from .frame_processor import FrameProcessor

        self.frame_processor: FrameProcessor = frame_processor
        self._last_frame = None
        self._frame_lock = threading.Lock()
        self._frame_consumer_running = False
        self._frame_consumer_task: asyncio.Task | None = None

        # Wrap the FrameProcessor in a viewer-mode track + MediaRelay so
        # WebRTC viewers can subscribe without spawning a duplicate pipeline.
        self.video_track = VideoProcessingTrack(
            frame_processor=frame_processor,
            session_id=frame_processor.session_id,
        )
        self.relay = MediaRelay()
        # The "primary" subscription is what the internal frame consumer pulls
        # from. It also keeps the underlying track alive so recv() is driven
        # even when no external viewer is attached.
        self._primary_relay_track = self.relay.subscribe(self.video_track, buffered=False)

    def start_frame_consumer(self):
        """Start a background task that continuously pulls frames to keep the
        pipeline moving and caches the latest one for capture_frame."""
        if self._frame_consumer_running:
            return
        self._frame_consumer_running = True
        self._frame_consumer_task = asyncio.create_task(self._consume_frames())

    async def _consume_frames(self):
        """Pull frames via the relay so MediaRelay drives recv() on the source
        track. Cache the latest VideoFrame for the capture_frame API."""
        while self._frame_consumer_running and self.frame_processor.running:
            try:
                frame = await self._primary_relay_track.recv()
            except Exception as e:
                logger.debug(f"Headless frame consumer recv() exited: {e}")
                break
            if frame is not None:
                with self._frame_lock:
                    self._last_frame = frame

    async def close(self):
        """Stop the frame processor and consumer."""
        self._frame_consumer_running = False
        if self._frame_consumer_task is not None:
            self._frame_consumer_task.cancel()
            try:
                await self._frame_consumer_task
            except asyncio.CancelledError:
                pass
        # Stop the viewer-mode track first; it does NOT own the FrameProcessor
        # so this is safe.
        try:
            await self.video_track.stop()
        except Exception as e:
            logger.debug(f"Error stopping headless video_track: {e}")
        self.frame_processor.stop()
        logger.info("Headless session closed")

    def get_last_frame(self):
        """Return the most recently cached frame, or None."""
        with self._frame_lock:
            return self._last_frame

    def __str__(self):
        return f"HeadlessSession(running={self.frame_processor.running})"
