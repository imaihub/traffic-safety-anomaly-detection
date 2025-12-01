from abc import ABC


class VideoWriterBase(ABC):
    def write(self, frame):
        pass

    def close(self):
        pass
