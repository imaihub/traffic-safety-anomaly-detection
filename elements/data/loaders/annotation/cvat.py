import os

from elements.data.datatypes.annotations.boundingbox import BoundingBox
from elements.data.datatypes.annotations.keypoint import Keypoint


class CVATAnnotations:
    def __init__(self, save_path: str, task_name: str = None, img_count: int = 1, bbox_class_names: list[str] = None, keypoint_class_names: list[str] = None, tag_class_names: list[str] = None, individual: bool = False):
        """
        Class for saving annotations to CVAT's Images 1.1 format

        :param task_name: name of the task
        :param img_count: number of images
        :param bbox_class_names: list of bounding box class names.
        :param keypoint_class_names: list of keypoint class names.
        :param tag_class_names: list of tag class names.
        """
        self.save_path = save_path
        self.task_name = task_name
        self.img_count = img_count
        self.bbox_class_names = bbox_class_names
        self.keypoint_class_names = keypoint_class_names
        self.tag_class_names = tag_class_names

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def initialize(self):
        """
        Writer header of xml file to savepath
        """
        header = f"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n" \
                 f"<annotations>\n" \
                 f"  <version>1.1</version>\n" \
                 f"  <meta>\n" \
                 f"  <job>\n" \
                 f"    <id>0</id>\n" \
                 f"    <name>{self.task_name}</name>\n" \
                 f"    <size>{self.img_count}</size>\n" \
                 f"    <mode>annotation</mode>\n" \
                 f"    <overlap>0</overlap>\n" \
                 f"    <bugtracker></bugtracker>\n" \
                 f"    <created></created>\n" \
                 f"    <updated></updated>\n" \
                 f"    <subset>default</subset>\n" \
                 f"    <start_frame>0</start_frame>\n" \
                 f"    <stop_frame>{self.img_count}</stop_frame>\n" \
                 f"    <frame_filter></frame_filter>\n" \
                 f"    <segments></segments>\n" \
                 f"    <owner>\n" \
                 f"      <username>cvds</username>\n" \
                 f"      <email>nhlcomputervision@gmail.com</email>\n" \
                 f"    </owner>\n" \
                 f"    <assignee></assignee>\n" \
                 f"    <labels>\n"
        if self.bbox_class_names is not None:
            for class_name in self.bbox_class_names:
                header += f"      <label>\n" \
                          f"        <name>{class_name}</name>\n" \
                          f"        <color></color>\n" \
                          f"        <type>rectangle</type>\n" \
                          f"        <attributes></attributes>\n" \
                          f"      </label>\n"
        if self.keypoint_class_names is not None:
            for class_name in self.keypoint_class_names:
                header += f"      <label>\n" \
                          f"        <name>{class_name}</name>\n" \
                          f"        <color></color>\n" \
                          f"        <type>points</type>\n" \
                          f"        <attributes></attributes>\n" \
                          f"      </label>\n"
        header += f"      </labels>\n" \
                  f"    </job>\n" \
                  f"    <dumped></dumped>\n" \
                  f"  </meta>\n"
        with open(self.save_path, 'w') as fp:
            fp.write(header)
        return self

    def __enter__(self):
        self.initialize()
        return self

    def save_bboxes(self, bboxes: list[BoundingBox], image_id: int, image_fname: str, image_height: int, image_width: int, is_new_entry: bool = True, clip_coordinates: bool = False):
        """
        Save bounding boxes to CVAT Images 1.1 xml format.

        :param bboxes: [[y1, x1, y2, x2, class id]]
        :param image_id: id of the image
        :param image_fname: image filename
        :param image_height: height of the image
        :param image_width: width of the image
        :param is_new_entry: if True will create new image id in xml file
        :param clip_coordinates: if True, clip coordinates to image boundaries
        """
        content = ""
        if is_new_entry:
            content = f"  <image id=\"{image_id}\" name=\"{image_fname}\" width=\"{image_width}\" height=\"{image_height}\">\n"

        for bbox in bboxes:
            class_name = bbox.get_class_name(self.bbox_class_names)
            xtl, ytl, xbr, ybr = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            if clip_coordinates:
                if xtl < 0:
                    xtl = 0
                if ytl < 0:
                    ytl = 0

                if xbr > image_width:
                    xbr = image_width
                if ybr > image_height:
                    ybr = image_height
            content += f"    <box label=\"{class_name}\" source=\"automatic\" xtl=\"{xtl}\" ytl=\"{ytl}\" xbr=\"{xbr}\" ybr=\"{ybr}\" z_order=\"0\"></box>\n"
        content += f"  </image>\n"
        with open(self.save_path, 'a') as fp:
            fp.write(content)

    def save_keypoints(self, keypoints: list[Keypoint], image_id: int, image_fname: str, image_height: int, image_width: int, is_new_entry: bool = True):
        """
        Save keypoints to CVAT Images 1.1 xml format

        :param keypoints: keypoints in [y, x, class id] format
        :param image_id: id of the image
        :param image_fname: image filename
        :param image_height: height of the image
        :param image_width: width of the image
        :param is_new_entry: if True will create new image id in xml file
        """
        content = ""
        if is_new_entry:
            content = f"  <image id=\"{image_id}\" name=\"{image_fname}\" width=\"{image_width}\" height=\"{image_height}\">\n"

        for kpt in keypoints:
            class_name = self.keypoint_class_names[int(kpt.get_class_id())]
            coords = f"{kpt.x},{kpt.y}"
            content += f"    <points label=\"{class_name}\" occluded=\"0\" source=\"automatic\" points=\"{coords}\" z_order=\"0\"></points>\n"
        content += f"  </image>\n"
        with open(self.save_path, 'a') as fp:
            fp.write(content)

    def close(self):
        footer = "</annotations>\n"
        with open(self.save_path, 'a') as fp:
            fp.write(footer)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
