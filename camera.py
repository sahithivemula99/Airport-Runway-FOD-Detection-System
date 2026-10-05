"""
Airport Runway Foreign Object Debris (FOD) Detection and Risk Assessment System
Threaded Video Stream & Camera Capture Module
"""

import time
import threading
from queue import Queue, Empty
from typing import Union, Tuple, Optional
import cv2
import numpy as np


class ThreadedCamera:
    """
    High-performance threaded camera and RTSP stream capture handler.
    Eliminates frame buffering lag by asynchronously fetching frames in a background thread.
    """

    def __init__(
        self, source: Union[int, str] = 0, queue_size: int = 2
    ) -> None:
        """
        Initialize camera stream capture thread.

        :param source: Camera index (0, 1) or stream URL ('rtsp://...', 'http://...').
        :param queue_size: Internal queue buffer size.
        """
        self.source: Union[int, str] = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.queue: Queue = Queue(maxsize=queue_size)
        self.stopped: bool = False
        self.is_opened: bool = False
        self.thread: Optional[threading.Thread] = None

        self._start_capture()

    def _start_capture(self) -> None:
        """Initialize OpenCV VideoCapture and launch background reading thread."""
        try:
            # Handle numeric camera indices vs string stream URLs
            if isinstance(self.source, str) and self.source.isdigit():
                self.source = int(self.source)

            self.cap = cv2.VideoCapture(self.source)

            # Optimize OpenCV VideoCapture properties if webcam
            if isinstance(self.source, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                print(f"[ThreadedCamera] Error: Unable to open camera source '{self.source}'.")
                self.is_opened = False
                return

            self.is_opened = True
            self.stopped = False
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            print(f"[ThreadedCamera] Stream started for source '{self.source}'.")
        except Exception as e:
            print(f"[ThreadedCamera] Exception opening source '{self.source}': {e}")
            self.is_opened = False

    def _update_loop(self) -> None:
        """Background thread loop that continuously reads frames from stream."""
        while not self.stopped and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Clear queue if full to keep only the freshest frame
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass

            self.queue.put(frame)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the latest available frame from queue buffer.

        :return: Tuple of (success_boolean, OpenCV_image_frame).
        """
        if not self.is_opened:
            return False, None

        try:
            frame = self.queue.get(timeout=0.2)
            return True, frame
        except Empty:
            return False, None

    def stop(self) -> None:
        """Stop background capture thread and release video capture hardware."""
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.is_opened = False
        print(f"[ThreadedCamera] Stream stopped for source '{self.source}'.")

    def __del__(self) -> None:
        """Destructor to ensure camera resources are released."""
        self.stop()
