from enum import Enum
from typing import Iterator, Optional, Tuple

from elements.data.datatypes.samplecontainer import SampleContainer
from elements.data.loaders.image.utils.imagecodecs import load_image
from elements.data.readers.base import BaseReader
from elements.data.samplers.video.base import BaseVideoSampler
from elements.data.transforms.preprocess.image import Compose


class FrameMode(Enum):
    SINGLE = 1
    PAIRS = 2


class FrameProcessor:
    def __init__(self, reader: BaseReader, transforms: Optional[Compose] = None, sampler: Optional[BaseVideoSampler] = None):
        """
        Args:
            reader: BaseReader-like (VideoReader, FolderReader, WebcamReader, etc.)
            transforms: callable (e.g. Compose of transforms)
            sampler: optional sampler (for training sequences)
        """
        self.reader = reader
        self.transforms = transforms or (lambda x: x)
        self.sampler = sampler

        # Metadata
        self.total_frames = getattr(reader, "total_frames", None)
        self.fps = getattr(reader, "fps", None)

    def get_shape(self) -> Tuple[int, int]:
        sample = self.__getitem__(0)
        if isinstance(sample, SampleContainer):
            return sample.org_width, sample.org_height
        elif isinstance(sample, list):
            return sample[0].org_width, sample[1].org_height
        else:
            raise NotImplementedError

    def __len__(self):
        return self.total_frames or 0

    def __getitem__(self, index: int):
        if self.sampler:
            frame_video_idx = getattr(self.reader, "frame_video_idx", list(range(self.total_frames)))
            indices = self.sampler.sample_indices(index, self.total_frames, frame_video_idx)
        else:
            indices = [index]

        scs = []
        for idx in indices:
            if not hasattr(self.reader, "get_frame"):
                raise NotImplementedError("Reader does not support indexing; use frames() for streaming sources")
            img, sc = self.reader.get_frame(idx)
            sc.org_width = img.shape[1]
            sc.org_height = img.shape[0]
            sc.org_image = img

            transformed_img = self.transforms(img)
            sc.image_data = transformed_img

            scs.append(sc)
        return scs if len(scs) > 1 else scs[0]

    def frames(self, skip_frames: int = 0) -> Iterator[tuple[int, list[SampleContainer]]]:
        """
        Streaming generator for live or sequential data. Yields SampleContainer instances
        wrapping the image and metadata.
        """
        for idx, img, sc in self.reader.frames(skip_frames):
            if self.sampler:
                # get sequence indices for this current idx
                frame_video_idx = getattr(self.reader, "frame_video_idx", list(range(self.total_frames)))
                indices = self.sampler.sample_indices(idx, self.total_frames, frame_video_idx)
                sc_sequence = []
                for i in indices:
                    img_i, sc_i = self.reader.get_frame(i)
                    sc_i.org_image = img_i
                    sc_i.org_width = img_i.shape[1]
                    sc_i.org_height = img_i.shape[0]
                    if self.transforms is not None:
                        sc_i.image_data = self.transforms(img_i)
                    else:
                        sc_i.image_data = img_i
                    sc_sequence.append(sc_i)
                yield idx, sc_sequence if len(sc_sequence) > 1 else sc_sequence[0]
            else:
                sc.org_image = img
                sc.org_width = img.shape[1]
                sc.org_height = img.shape[0]
                if self.transforms is not None:
                    sc.image_data = self.transforms(img)
                else:
                    sc.image_data = img
                yield idx, [sc]

    def __call__(self, img):
        if isinstance(img, str):
            img = load_image(img)
        return self.transforms(img)

    def release(self):
        self.reader.release()
