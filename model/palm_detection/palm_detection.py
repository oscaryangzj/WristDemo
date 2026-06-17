import copy
from typing import (
    Tuple,
    Optional,
    List,
)
from math import (
    sin,
    cos,
    atan2,
    pi,
)
import onnxruntime
import numpy as np
from utils.utils import (
    normalize_radians,
    keep_aspect_resize_and_pad,
)

class PalmDetection:
    def __init__ (
        self,
        model_path: Optional[str] = 'model/palm_detection/palm_detection_full_inf_post_192x192.onnx',
        score_threshold: Optional[float] = 0.60,
        providers: Optional[List] = ['CUDAExecutionProvider', 'CPUExecutionProvider'],
    ) :
        self.score_threshold = score_threshold

        session_option = onnxruntime.SessionOptions ()
        session_option.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession (
            model_path,
            sess_options = session_option,
            providers = providers
        )
        self.providers = self.onnx_session.get_providers ()

        self.input_shapes = [
            input.shape for input in self.onnx_session.get_inputs ()
        ]
        self.input_names = [
            input.name for input in self.onnx_session.get_inputs ()
        ]
        self.output_names = [
            output.name for output in self.onnx_session.get_outputs ()
        ]
        self.square_standard_size = 0

    def __call__ (self, image: np.ndarray) -> np.ndarray : 
        temp_image = copy.deepcopy (image)

        # preprocess
        inference_image = self.__preprocess (temp_image)

        # inference
        inference_image = np.asarray([inference_image], dtype = np.float32)
        boxes = self.onnx_session.run (
            self.output_names,
            {input_name : inference_image for input_name in self.input_names}
        )

        # postprocess
        hands = self.__postprocess (
            image = temp_image,
            boxes = boxes[0]
        )
        return hands

    def __preprocess (
        self,
        image: np.ndarray,
        swap: Optional[Tuple[int, int, int]] = (2, 0, 1)
    ) -> np.ndarray :
        input_h = self.input_shapes[0][2]
        input_w = self.input_shapes[0][3]
        image_h, image_w = image.shape[: 2]

        self.square_standard_size = max (image_h, image_w)
        self.square_padding_half_size = abs (image_h - image_w) // 2
        padded_image, resized_image = keep_aspect_resize_and_pad (
            image = image,
            resize_width = input_w,
            resize_height = input_h
        )
        pad_size_half_h = max(0, (input_h - resized_image.shape[0]) // 2)
        pad_size_half_w = max(0, (input_w - resized_image.shape[1]) // 2)

        self.pad_size_scale_h = pad_size_half_h / input_h
        self.pad_size_scale_w = pad_size_half_w / input_w

        padded_image = np.divide (padded_image, 255.0)
        padded_image = padded_image [..., : : - 1]
        padded_image = padded_image.transpose(swap)
        padded_image = np.ascontiguousarray (
            padded_image,
            dtype=np.float32,
        )

        return padded_image

    def __postprocess (
        self,
        image: np.ndarray,
        boxes: np.ndarray
    ) -> np.ndarray :
        image_h = image.shape[0]
        image_w = image.shape[1]

        hands = []
        keep = boxes[:, 0] > self.score_threshold
        boxes = boxes[keep, :]

        for box in boxes:
            pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y = box
            if box_size > 0:
                kp02_x = kp2_x - kp0_x
                kp02_y = kp2_y - kp0_y
                sqn_rr_size = 2.9 * box_size
                rotation = 0.5 * pi - atan2(- kp02_y, kp02_x)
                rotation = normalize_radians(rotation)
                sqn_rr_center_x = box_x + 0.5 * box_size * sin(rotation)
                sqn_rr_center_y = box_y - 0.5 * box_size * cos(rotation)
                sqn_rr_center_y = (sqn_rr_center_y * self.square_standard_size - self.square_padding_half_size) / image_h
                hands.append (
                    [
                        sqn_rr_size,
                        rotation,
                        sqn_rr_center_x,
                        sqn_rr_center_y,
                    ]
                )

        return np.asarray (hands)
