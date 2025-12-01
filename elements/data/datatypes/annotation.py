from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional, Union, Any, Type
import numpy as np


class AnnotationBase(ABC):
    """
    Abstract Base Class for annotations.
    Inherit from this class when adding a new annotation type.
    """
    def __init__(self):
        """
        Note: all annotations should support calling their __init__ without parameters
        """
        super().__init__()

    def to_array(self) -> np.ndarray:
        """
        Used for storing into .npz file. Generate numpy array representation of this annotation.
        """
        pass


AnnotationInstance = Union[AnnotationBase, Any]  # 'Any' is only allowed for the generic dataloader from the post_transform stage
Annotations = Union[list[AnnotationInstance], AnnotationInstance]


class AnnotationDict:
    """
    Annotation dictionary class.
    This class keeps a dictionary of annotations where the key is an AnnotationBase type.
    The value can be either an AnnotationBase object (if enforce_annotationbase is True),
    or a mix of AnnotationBase objects and other types (if enforce_annotationbase is False).

    """
    def __init__(self):
        self._annotations = {}
        self._enforce_annotationbase = True

    def clear(self):
        """
        Clear annotations.
        """
        self._annotations.clear()

    def set_enforce_annotationbase(self, enforce: bool):
        """
        Set enforcement of annotationbase data.
        :param enforce: if True, annotations are enforced to be AnnotationBase objects.
        """
        self._enforce_annotationbase = enforce

    def has(self, annotation_type: Type[AnnotationBase], allow_empty: bool = False) -> bool:
        """
        Check if this sample has at least one annotation of the specified type.
        """
        if annotation_type not in self._annotations.keys():
            return False
        if allow_empty:
            return True
        annotations = self._annotations[annotation_type]
        return not isinstance(annotations, list) or len(annotations) > 0

    def get_types(self) -> list[Type[AnnotationBase]]:
        """
        Get all annotation types which have at least one annotation.
        """
        return [k for k in self._annotations.keys() if self.has(k)]

    def get_all_annotations(self) -> dict[Type[AnnotationBase], Annotations]:
        """
        Get all annotations.
        :return: dictionary with all annotations.
        """
        return self._annotations

    def get(self, annotation_type: Type[AnnotationBase], allow_empty: bool = False) -> Annotations:
        """
        Get annotations. If there are no annotations of the given type, None is returned.
        """
        if self.has(annotation_type, allow_empty):
            return self._annotations[annotation_type]
        else:
            return None

    def set(self, annotation_type: Type[AnnotationBase], annotations: Optional[Annotations]):
        """
        Set annotations. Annotations can be removed using None. Previous annotations of the given type will be overwritten.

        Note that annotations must be instances of AnnotationBase before the post_transform stage. This will be checked if self._enforce_annotationbase is set to True.
        """
        if annotations is None:
            self._annotations.pop(annotation_type, None)
        else:
            if self._enforce_annotationbase:
                annots = annotations
                if not isinstance(annots, list):
                    annots = [annots]
                assert all(isinstance(annot, AnnotationBase) for annot in annots)

            self._annotations[annotation_type] = annotations
