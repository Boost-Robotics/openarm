"""CAN bus message multiplexer for filtering messages by arbitration ID."""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections import defaultdict
from time import time

import can

logger = logging.getLogger(__name__)


class Bus:
    """CAN bus wrapper with automatic recovery from bus-off / ENOBUFS."""

    def __init__(self, bus: can.BusABC) -> None:
        """Initialize the bus multiplexer.

        Args:
            bus: The underlying CAN bus interface.

        """
        self.bus = bus
        self.lookup = defaultdict(list)
        self._channel: str | None = self._resolve_channel()
        self._consecutive_timeouts = 0
        self._timeout_reset_threshold = 3
        self._reset_cooldown_s = 2.0
        self._last_reset_ts = 0.0

    def _resolve_channel(self) -> str | None:
        """Extract the SocketCAN interface name (e.g. 'can3') from the bus."""
        if hasattr(self.bus, "channel"):
            return str(self.bus.channel)
        if hasattr(self.bus, "channel_info"):
            info = str(self.bus.channel_info)
            for token in info.split():
                if token.startswith("can"):
                    return token
        return None

    def _maybe_reset_on_timeouts(self) -> None:
        if self._consecutive_timeouts < self._timeout_reset_threshold:
            return
        now = time()
        if (now - self._last_reset_ts) < self._reset_cooldown_s:
            return
        logger.warning(
            "No RX on %s for %d consecutive recv() calls — resetting bus",
            self._channel,
            self._consecutive_timeouts,
        )
        if self.reset():
            self._last_reset_ts = now
        self._consecutive_timeouts = 0

    def reset(self) -> bool:
        """Bounce the CAN interface to recover from bus-off / TX overflow.

        Returns True if the reset succeeded.
        """
        ch = self._channel
        if ch is None:
            logger.warning("Cannot reset: unknown CAN channel")
            return False
        try:
            prefix = ["sudo"] if shutil.which("sudo") else []
            subprocess.run(
                prefix + ["ip", "link", "set", ch, "down"],
                check=True, timeout=5,
            )
            subprocess.run(
                prefix + ["ip", "link", "set", ch, "up",
                           "type", "can", "bitrate", "1000000"],
                check=True, timeout=5,
            )
            self.lookup.clear()
            self._consecutive_timeouts = 0
            logger.info("CAN interface %s reset successfully", ch)
            return True
        except Exception:
            logger.exception("Failed to reset CAN interface %s", ch)
            return False

    def send(self, msg: can.Message, timeout: float | None = None) -> None:
        """Send a CAN message, auto-recovering from ENOBUFS (error 105).

        Args:
            msg: The CAN message to send.
            timeout: Optional send timeout in seconds.

        """
        try:
            self.bus.send(msg, timeout)
        except can.CanOperationError as exc:
            if "105" in str(exc) or "No buffer space" in str(exc):
                logger.warning("TX buffer full on %s — resetting bus", self._channel)
                if self.reset():
                    self.bus.send(msg, timeout)  # retry once after reset
                else:
                    raise
            else:
                raise

    def recv(
        self, arbitration_id: int, timeout: float | None = None
    ) -> can.Message | None:
        """Receive a CAN message with the specified arbitration ID.

        Messages with other arbitration IDs are queued for later retrieval.

        Args:
            arbitration_id: The arbitration ID to filter for.
            timeout: Optional receive timeout in seconds. None means wait indefinitely.

        Returns:
            The received CAN message, or None if timeout occurred.

        """
        queue = self.lookup[arbitration_id]
        if len(queue) > 0:
            return queue.pop(0)

        if timeout is None:
            while True:
                msg = self.bus.recv()
                if msg.is_error_frame or not msg.is_rx:
                    continue
                if msg.arbitration_id == arbitration_id:
                    return msg
                self.lookup[msg.arbitration_id].append(msg)
        else:
            end = time() + timeout
            while timeout > 0:
                msg = self.bus.recv(timeout)
                if msg is None:
                    self._consecutive_timeouts += 1
                    self._maybe_reset_on_timeouts()
                    return None
                if msg.is_error_frame or not msg.is_rx:
                    continue
                if msg.arbitration_id == arbitration_id:
                    self._consecutive_timeouts = 0
                    return msg
                self.lookup[msg.arbitration_id].append(msg)
                timeout = end - time()

        return None
