from typing import Optional, Callable, Type, TypeVar, Union

import numpy as np

from elements.data.datatypes.annotation import AnnotationBase
from elements.data.datatypes.annotations.boundingbox import BoundingBox
from elements.data.datatypes.annotations.mask import SegmentationMask, ClassMask, InstanceMask, Mask
from elements.data.datatypes.samplecontainer import SampleContainer

AnnotationType = TypeVar('AnnotationType', bound=AnnotationBase)


class RemapLabels:
    """
    Remaps annotation label IDs based on the given mapping. Mapping behaviour is as follows:
    k: v        Remaps class id 'k' to 'v'
    k: None     Removes annotations with class id 'k' (not supported for all annotation types!)
    None: v     All class ids which do not have a mapping defined explicitly will map to 'v' instead

    :param mapping: the cuda device to training on
    :param strict: if True, an error will be raised if a class ID is encountered for which no mapping exists
    :param types: the annotation types for which the mapping should be applied. If not specified, all supported annotation types will be remapped.
    """
    def __init__(self, mapping: dict[Optional[Union[int, str]], Optional[int]], strict: bool = True, from_class_names: bool = False, types: Optional[list[Type[AnnotationBase]]] = None):
        self._mapping = mapping
        self._strict = strict
        self._from_class_names = from_class_names
        self._types = types
        self._relabel_fns: dict[Type[AnnotationType], Callable[[list[AnnotationType]], list[AnnotationType]]] = {
            BoundingBox: self._remap_boundingboxes,
            SegmentationMask: self._remap_masks,
            ClassMask: self._remap_masks,
            InstanceMask: self._remap_masks,
        }

        if self._types is not None:
            assert all([annot_type in self._relabel_fns for annot_type in self._types])

        if self._from_class_names:
            assert all(isinstance(m, str) for m in self._mapping.keys() if m is not None)
        else:
            assert all(isinstance(m, int) for m in self._mapping.keys() if m is not None)

            # get a mapping that has no None keys, or keys that map to themselves (needed for Masks)
            self._mapping_specific = {k: v for k, v in self._mapping.items() if k is not None and k != v}
            # integer np.ndarray of keys (needed for Masks)
            self._mapping_keys = np.array([int(x) for x in self._mapping.keys() if x is not None])

    def __call__(self, sample: SampleContainer) -> SampleContainer:
        for type, remapped_fn in self._relabel_fns.items():
            if self._types is not None and type not in self._types:
                continue

            annots = sample.annotations.get(type)
            if annots is None:
                continue

            is_single = not isinstance(annots, list)
            if is_single:
                annots = [annots]

            remapped_annots = remapped_fn(annots)

            if is_single:
                remapped_annots = remapped_annots[0]
            sample.annotations.set(type, remapped_annots)

        return sample

    def get_mapping(self):
        return self._mapping

    def _remap_boundingboxes(self, boxes: list[BoundingBox]) -> list[BoundingBox]:
        result = []
        for box in boxes:
            box_id = box.class_name if self._from_class_names else box.get_class_id()

            if box_id not in self._mapping.keys():
                if None in self._mapping.keys():
                    box_id = None
                elif self._strict:
                    raise KeyError(f"No mapping defined for class ID {box_id}")
                else:
                    result.append(box)
                    continue

            new_id = self._mapping[box_id]
            if new_id is None:
                continue

            box.set_class_id(new_id)
            result.append(box)
        return result

    def _remap_masks(self, masks: list[Mask]) -> list[Mask]:
        return [self._remap_mask(mask) for mask in masks]

    def _remap_mask(self, mask_annot: Mask) -> Mask:
        mask = mask_annot.mask

        # check if any IDs map to None (which is not allowed for masks)
        if any([v is None for v in self._mapping.values()]):
            raise ValueError(f"Mapping target can not be 'None' when remapping Masks")

        # check if there are any IDs in the mask which do not have a mapping
        if self._strict or None in self._mapping:
            isnone = np.isin(mask, self._mapping_keys, invert=True)

        # replace explicitly mapped values
        # todo: this is slow for large _mapping_specific dict
        def map(x):
            return self._mapping_specific.get(x, x)

        mask = np.vectorize(map)(mask)

        # handle values without mapping
        if None in self._mapping:
            # apply the default 'None' mapping
            mask[isnone] = self._mapping[None]
        elif np.any(isnone):
            # one or more IDs don't have a mapping specified
            missing_mappings = np.unique(mask[isnone])
            raise KeyError(f"No mapping defined for class IDs {missing_mappings}")

        mask_annot.mask = mask
        return mask_annot
