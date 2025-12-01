import copy
import os
from typing import Optional, Any, Union, Type

import torch

from elements.data.datatypes.annotation import AnnotationDict, AnnotationBase
from elements.data.datatypes.cachedresource import PersistState, CachedResource


class SampleContainer:
    """
    SampleContainer object containing all information of one sample.

    Attributes:

    :param image_fname: the filename of the sample.
    :param image_fpath: the absolute filepath of the sample.
    :param image_data: image data stored in a CachedResource object.
    :param class_id: class id of the sample (might be moved to the annotations object)
    :param class_name: class name of the sample (might be moved to the annotations object)
    :param annotations_fpaths: paths to the annotations
    :param annotations: AnnotationDict containing the annotations. The annotations can be accessed by passing the AnnotationType to the get function.
    :param org_width: original width of the sample (useful for reconstructing the image from fixed tiles)
    :param org_height: original height of the sample (useful for reconstructing the image from fixed tiles)
    :param loaded: boolean to check whether the image data has been loaded.
    :param preprocessed: boolean to check whether the image data has been pre-processed already.
    :param default_persist: persistence state of the sample. If True, keeps the data in memory, otherwise store data in cache file.
    :param cache_dir: directory where the cache files should be stored.
    """
    def __init__(self):
        self._image_fname: Optional[Union[list[str], str]] = None
        self._image_fpath: Optional[Union[list[str], str]] = None
        self._image_data: Optional[CachedResource] = None
        self._org_image: Optional[CachedResource] = None

        self._class_id: Optional[int] = None
        self._class_name: Optional[str] = None

        self._annotations_fpaths: dict[Type[AnnotationBase], Union[str, list[str]]] = {}
        self._annotations: Union[AnnotationDict, list[AnnotationDict]] = AnnotationDict()  # should only be a List after collation

        self._org_width: Optional[Union[int, list[int]]] = None
        self._org_height: Optional[Union[int, list[int]]] = None

        self._meta_data: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = dict()

        self._loaded: bool = False
        self._preprocessed: bool = False
        self._default_persist: PersistState = PersistState(state=True)  # main persistence state
        self._cache_dir: Union[str, None] = None

    def clear_data(self):
        """
        Clear all data (keeping paths), i.e. restore to 'unloaded' state
        """
        self._image_data = None
        self._annotations.clear()
        self._org_width = None
        self._org_height = None
        self._meta_data.clear()
        self._loaded = False
        self._preprocessed = False

        # might be set by dataset info directly, so don't clear these (won't take up much memory anyway)
        # self._class_id = None
        # self._class_name = None

    def save_cached_resources(self):
        """
        Save CachedResource data.
        """
        if not self.default_persist.state:
            for attr_name, value in self.__dict__.items():
                if isinstance(value, CachedResource):
                    value.save_cache_file()

    def reload_cached_resources(self):
        """
        Reload CachedResource data
        """
        for attr_name, value in self.__dict__.items():
            if isinstance(value, CachedResource):
                value()

    def free_cached_resources(self):
        """
        Free CachedResources if sample not persistent.
        """
        if not self.default_persist.state:
            for attr_name, value in self.__dict__.items():
                # only clear cached resources if not persistent
                if isinstance(value, CachedResource):
                    value.persist = self.default_persist.state
                    value.free()

    def get_cached_resource_attr_names(self):
        """
        Get attribute names of CachedResource objects.
        :return: list of attribute names
        """
        attr_names = []
        for attr_name, value in self.__dict__.items():
            if isinstance(value, CachedResource):
                attr_names.append(attr_name)
        return attr_names

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Free memory upon exit call of context manager.
        TODO: could also be implemented such that memory is also freed when usage is too high.
        :param exc_type: 
        :param exc_val: 
        :param exc_tb: 
        :return: 
        """
        self.free_cached_resources()

    @property
    def image_fname(self):
        return self._image_fname

    @image_fname.setter
    def image_fname(self, image_fname):
        self._image_fname = image_fname

    @property
    def image_fpath(self) -> Union[str, list[str]]:
        return self._image_fpath

    @image_fpath.setter
    def image_fpath(self, image_fpath: Union[str, list[str]]):
        self._image_fpath = image_fpath
        if isinstance(image_fpath, str):
            self._image_fname = os.path.basename(self._image_fpath)
        else:
            self._image_fname = [os.path.basename(path) for path in self._image_fpath]

    @property
    def image_data(self) -> CachedResource:
        if self._image_data is not None:
            self._image_data()
        return self._image_data

    @image_data.setter
    def image_data(self, image_data: Any):
        if image_data is not None:
            if not isinstance(self._image_data, CachedResource):
                self._image_data = CachedResource(data=image_data, name=self._image_fname, persist=self.default_persist.state, cache_dir=self._cache_dir)
            else:
                self._image_data.data = image_data

    @property
    def org_image(self) -> CachedResource:
        if self._org_image is not None:
            self._org_image()
        return self._org_image

    @org_image.setter
    def org_image(self, image_data: Any):
        if image_data is not None:
            if not isinstance(self._org_image, CachedResource):
                self._org_image = CachedResource(data=image_data, name=self._image_fname, persist=self.default_persist.state, cache_dir=self._cache_dir)
            else:
                self._org_image.data = image_data

    @property
    def class_id(self) -> Union[int, list[int], torch.Tensor]:
        return self._class_id

    @class_id.setter
    def class_id(self, class_id: Union[int, list[int], torch.Tensor]):
        self._class_id = class_id

    @property
    def class_name(self) -> Union[str, list]:
        return self._class_name

    @class_name.setter
    def class_name(self, class_name: Union[str, list[str]]):
        self._class_name = class_name

    @property
    def annotations_fpaths(self) -> dict[Type[AnnotationBase], Union[str, list[str]]]:
        return self._annotations_fpaths

    @annotations_fpaths.setter
    def annotations_fpaths(self, annotations_fpaths: dict[Type[AnnotationBase], Union[str, list[str]]]):
        self._annotations_fpaths = annotations_fpaths

    @property
    def annotations(self) -> AnnotationDict | list[AnnotationDict]:
        return self._annotations

    @annotations.setter
    def annotations(self, annotations: AnnotationDict | list[AnnotationDict]):
        self._annotations = annotations

    @property
    def org_height(self) -> Union[int, list[int]]:
        return self._org_height

    @org_height.setter
    def org_height(self, org_height: Union[int, list[int]]):
        self._org_height = org_height

    @property
    def org_width(self) -> Union[int, list[int]]:
        return self._org_width

    @org_width.setter
    def org_width(self, org_width: Union[int, list[int]]):
        self._org_width = org_width

    @property
    def meta_data(self) -> Union[dict[str, Any], list[dict[str, Any]]]:
        return self._meta_data

    @meta_data.setter
    def meta_data(self, meta_data: Union[dict[str, Any], list[dict[str, Any]]]):
        self._meta_data = meta_data

    @property
    def loaded(self) -> bool:
        return self._loaded

    @loaded.setter
    def loaded(self, loaded: bool):
        self._loaded = loaded

    @property
    def default_persist(self) -> PersistState:
        return self._default_persist

    @default_persist.setter
    def default_persist(self, default_persist: bool):
        self._default_persist = PersistState(state=default_persist)

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, cache_dir: str):
        self._cache_dir = cache_dir


Sample = Union[SampleContainer, list[SampleContainer]]
