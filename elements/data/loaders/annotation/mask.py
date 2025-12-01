import os
from abc import ABC
from typing import Union

import numpy as np

from elements.data.datatypes.annotations.mask import ClassMask, InstanceMask
from elements.data.datatypes.samplecontainer import SampleContainer
from elements.data.loaders.annotation.utils.pascal_voc import load_class_mask_pascal
from elements.data.loaders.annotation.utils.supervisely import load_svly_as_class_ids, load_svly_as_inst_ids
from elements.data.loaders.image.utils.imagecodecs import load_mono_image, load_image
from elements.data.loaders.image.utils.numpy import load_npy


class LoadClassMasksFileSC:
    """
    Handler to load masks containing class ids (each pixel represents a class id) into a SampleContainer.
    """

    loaders = {'json': load_svly_as_class_ids, 'png': load_mono_image}

    def __init__(self, scale: int = None):
        self.scale = scale

    def get_types(self) -> list[str]:
        """
        Returns the file types supported by this object.

        :returns: a list of strings representing the file extensions (including the leading period) supported by this object.
        :rtype: list of str
        """
        return list(self.loaders.keys())

    def __call__(self, sample: SampleContainer) -> SampleContainer:
        """
        Loads a class mask into a `SampleContainer`.

        This method loads a class mask from the `annotations_fpaths` attribute of the input `SampleContainer`, using
        the appropriate loader based on the file extension. If the class mask is a tuple containing class IDs and names,
        the `class_id` and `class_name` attributes of the `SampleContainer` are updated accordingly.

        :param sample: a `SampleContainer` object.
        :type sample: SampleContainer
        :returns: the modified `SampleContainer` object, with the class mask loaded and stored in the `annotations` attribute.
        :rtype: SampleContainer
        :raises ValueError: If the input `SampleContainer` does not have `class_mask_fpath` information.
        """
        class_mask_fpath = sample.annotations_fpaths.get(ClassMask, None)
        if class_mask_fpath is None:
            raise ValueError(f"Cannot execute {type(self).__name__}: sample does not have class_mask_fpath info.")
        ext = os.path.splitext(class_mask_fpath)[1][1:]
        class_mask = self.loaders[ext](class_mask_fpath)
        if self.scale:
            class_mask *= self.scale

        if isinstance(class_mask, tuple):
            class_mask, class_ids_names = class_mask
            sample.class_id = list(class_ids_names.keys())
            sample.class_name = list(class_ids_names.values())

        sample.annotations.set(ClassMask, ClassMask(class_mask))
        return sample


class ABCLoadInstanceMasksSC(ABC):
    """
    This abstract base class defines the interface for loading instance masks.
    It has three loaders for png, npy, and json file formats.
    """
    loaders = {'png': lambda x: [load_mono_image(x), None, None], 'npy': lambda x: [load_npy(x), None, None], 'json': load_svly_as_inst_ids}

    def get_types(self):
        """Returns the file types supported by this object.

        :returns: a list of strings representing the file extensions (including the leading period) supported by this object.
        :rtype: list of str
        """
        return self.loaders.keys()

    def __call__(self, inst_mask_fpath: Union[list, str]):
        """
        Loads the instance masks from the given file path.
        If the file path is None, it raises a ValueError.

        :param inst_mask_fpath: a file path to load instance masks from.
        :type inst_mask_fpath: str
        :raises ValueError: If `inst_mask_fpath` is None.
        """
        if inst_mask_fpath is None:
            raise ValueError(f"Cannot execute {type(self).__name__}: sample does not have inst_mask_fpath info.")


class LoadInstanceMasksFileSC(ABCLoadInstanceMasksSC):
    """
    This class inherits from the ABCLoadInstanceMasksSC abstract base class and implements the loading of instance masks from a file path.
    It loads the instance masks from a SampleContainer object's annotations_fpaths dictionary using the InstanceMask key.
    """
    def __call__(self, sample: SampleContainer) -> SampleContainer:
        """
        Loads the instance masks from the `SampleContainer` object's `annotations_fpaths` dictionary using the
        `InstanceMask` key. It then sets the `class_id` and `class_name` attributes of the `SampleContainer` object and returns it.

        :param sample: a `SampleContainer` object containing the annotation file paths.
        :type sample: SampleContainer
        :return: a `SampleContainer` object containing the instance masks.
        :rtype: SampleContainer
        """
        inst_mask_fpath = sample.annotations_fpaths.get(InstanceMask, None)
        super().__call__(inst_mask_fpath=inst_mask_fpath)

        ext = os.path.splitext(inst_mask_fpath)[1][1:]
        instance_ids, class_ids, class_name = self.loaders[ext](inst_mask_fpath)
        sample.class_id = class_ids
        sample.class_name = class_name

        sample.annotations.set(InstanceMask, InstanceMask(instance_ids))
        return sample


class LoadInstanceMasksFolderSC(ABCLoadInstanceMasksSC):
    def __call__(self, sample: SampleContainer) -> SampleContainer:
        inst_mask_fpaths = sample.annotations_fpaths.get(InstanceMask, None)
        super().__call__(inst_mask_fpath=inst_mask_fpaths)

        if not isinstance(inst_mask_fpaths, list):
            raise ValueError(f"Instance masks fpaths is not a list of paths, maybe you want to use LoadInstanceMasksFileSC instead?")

        instance_ids, class_ids, class_names = np.array([]), None, None

        # if instance mask consists of multiple separate masks
        for i, fp in enumerate(inst_mask_fpaths):
            ext = os.path.splitext(fp)[1][1:]
            mask, class_ids, class_names = self.loaders[ext](fp)
            mask[mask > 0] = i
            if instance_ids.size == 0:
                instance_ids = mask
            else:
                instance_ids += mask

        if class_ids is not None:
            sample.class_id = class_ids
        if class_names is not None:
            sample.class_names = class_names

        sample.annotations.set(InstanceMask, InstanceMask(instance_ids))
        return sample


class LoadPascalVOCClassMaskFileSC:
    loaders = {'png': load_class_mask_pascal}

    def __init__(self, label_map: dict = None):
        if label_map is None:
            raise ValueError(f"Cannot execute {type(self).__name__}: sample does not have labelmap_path info.")

        self._label_map = label_map

    def get_types(self):
        return self.loaders.keys()

    def __call__(self, sample: SampleContainer) -> SampleContainer:
        segm_mask_fpath = sample.annotations_fpaths.get(ClassMask, None)

        if segm_mask_fpath is None:
            raise ValueError(f"Cannot execute {type(self).__name__}: sample does not have segm_mask_fpath info.")
        ext = os.path.splitext(segm_mask_fpath)[1][1:]
        class_mask = self.loaders[ext](segm_mask_fpath, self._label_map)
        sample.annotations.set(ClassMask, ClassMask(class_mask))

        return sample


class LoadPascalVOCInstanceMaskFileSC:
    loaders = {'png': load_image}

    def get_types(self):
        return self.loaders.keys()

    def __call__(self, sample: SampleContainer) -> SampleContainer:
        segm_mask_fpath = sample.annotations_fpaths.get(InstanceMask, None)

        if segm_mask_fpath is None:
            raise ValueError(f"Cannot execute {type(self).__name__}: sample does not have segm_mask_fpath info.")
        ext = os.path.splitext(segm_mask_fpath)[1][1:]
        instance_mask = self.loaders[ext](segm_mask_fpath)
        sample.annotations.set(InstanceMask, InstanceMask(instance_mask))

        return sample
