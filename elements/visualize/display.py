import os
from enum import Enum

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

try:
    matplotlib.use("TkAgg")
except:
    pass


def figure_size_from_pixels(width_px: int, height_px: int, dpi: int = 100) -> tuple[float, float]:
    """
    Convert pixel dimensions to a Matplotlib figure size in inches.

    :param width_px: Width of the figure in pixels.
    :param height_px: Height of the figure in pixels.
    :param dpi: Dots per inch (resolution). Default is 100.

    :returns: (width_in, height_in) suitable for use in plt.figure(figsize=..., dpi=...).
    """
    return width_px / dpi, height_px / dpi


class DisplayStrategy(Enum):
    AUTO = -1
    CV2 = 0
    MATPLOTLIB = 1


class Display:
    """
    This class enables functionality around visualizing numpy arrays with cv2.imshow or matplotlib,
    with automatic fallback if display is not available.
    """
    def __init__(self, window_name: str = "data", display_strategy: DisplayStrategy = DisplayStrategy.AUTO, enabled: bool = True) -> None:
        self.window_name = window_name
        self.enabled = enabled
        if not self.enabled:
            return

        self.fig, self.ax = None, None
        self.figure_size = None
        self.plot_image = None

        if display_strategy == DisplayStrategy.AUTO:
            if self.can_use_cv2():
                self.display_strategy = DisplayStrategy.CV2
            else:
                print("[Display] cv2 GUI not available, falling back to matplotlib.")
                self.display_strategy = DisplayStrategy.MATPLOTLIB
        elif display_strategy == DisplayStrategy.CV2:
            if self.can_use_cv2():
                self.display_strategy = DisplayStrategy.CV2
            else:
                print("[Display] cv2 GUI not available, falling back to matplotlib.")
                self.display_strategy = DisplayStrategy.MATPLOTLIB
        else:
            self.display_strategy = DisplayStrategy.MATPLOTLIB

        if self.display_strategy == DisplayStrategy.CV2:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    @staticmethod
    def can_use_cv2() -> bool:
        """
        Safely check if OpenCV GUI (cv2.imshow) is likely to work.
        """
        # If on linux (posix), check for Display environment variable
        if os.name == "posix" and not os.environ.get("DISPLAY"):
            return False
        return True

    def show_image(self, image: np.ndarray) -> None:
        """
        Shows the image using either cv2 or matplotlib, depending on environment.
        Press 'q' to close if using cv2.
        Expects images in RGB format, converts it automatically to BGR so cv2 can show it as RGB
        """
        if not self.enabled:
            return

        if self.display_strategy == DisplayStrategy.CV2:
            cv2.imshow(self.window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(10) & 0xff == ord('q'):
                self.close()
        elif self.display_strategy == DisplayStrategy.MATPLOTLIB:
            if self.fig is None or self.ax is None:
                self.figure_size = figure_size_from_pixels(width_px=image.shape[1], height_px=image.shape[0], dpi=100)
                self.fig, self.ax = plt.subplots(figsize=self.figure_size, dpi=100)
                self.plot_image = self.ax.imshow(image)
                self.ax.axis('off')
                plt.ion()
                plt.show(block=False)

            self.plot_image.set_data(image)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            raise NotImplementedError("Selected Display Strategy not implemented.")

    def set_title(self, title: str):
        """
        Set visual title for additional information not included in the image, like shown image path
        """
        if not self.enabled:
            return

        if self.display_strategy == DisplayStrategy.CV2:
            cv2.setWindowTitle(self.window_name, title)
        elif self.display_strategy == DisplayStrategy.MATPLOTLIB and self.ax is not None:
            self.ax.set_title(title)
            self.fig.canvas.flush_events()

    def close(self) -> None:
        """
        Closes all OpenCV windows if not in remote mode.
        """
        if not self.enabled:
            return

        if self.display_strategy == DisplayStrategy.CV2:
            cv2.destroyAllWindows()
        elif self.display_strategy == DisplayStrategy.MATPLOTLIB:
            if self.fig is not None:
                plt.close(self.fig)

    def __enter__(self) -> "Display":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
