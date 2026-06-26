#!/usr/bin/env python

import argparse
from math import degrees

import cv2 as cv
import numpy as np

from model import HandLandmark, PalmDetection
from utils import CvFpsCalc
from utils.utils import rotate_and_crop_rectangle


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=int, default=0)
    parser.add_argument('-wi', '--width', type=int, default=640)
    parser.add_argument('-he', '--height', type=int, default=480)
    parser.add_argument('-mdc', '--min_detection_confidence', type=float, default=0.6)
    parser.add_argument('-mlc', '--min_landmark_confidence', type=float, default=0.5)
    parser.add_argument('-dif', '--disable_image_flip', action='store_true')
    parser.add_argument(
        '--providers',
        choices=['default', 'cpu', 'coreml'],
        default='default',
        help='ONNX Runtime providers to use.',
    )
    return parser.parse_args()


def build_providers(provider_mode):
    if provider_mode == 'cpu':
        return ['CPUExecutionProvider']
    if provider_mode == 'coreml':
        return [
            (
                'CoreMLExecutionProvider',
                {
                    'ModelFormat': 'MLProgram',
                    'MLComputeUnits': 'ALL',
                },
            ),
            'CPUExecutionProvider',
        ]
    return None


def hands_to_rects(image, hands):
    image_height, image_width = image.shape[:2]
    wh_ratio = image_width / image_height
    rects = []

    for hand in hands:
        sqn_rr_size, rotation, center_x, center_y = hand
        cx = center_x * image_width
        cy = center_y * image_height
        rect_width = sqn_rr_size * image_width
        rect_height = sqn_rr_size * wh_ratio * image_height
        rects.append([cx, cy, rect_width, rect_height, degrees(rotation)])

    return np.asarray(rects, dtype=np.float32)


def draw_handboxes(image, rects):
    for index, rect in enumerate(rects, start=1):
        cx, cy, rect_width, rect_height, degree = rect
        box = cv.boxPoints(((cx, cy), (rect_width, rect_height), degree))
        box = box.astype(np.intp)

        cv.drawContours(image, [box], 0, (0, 0, 255), 2, cv.LINE_AA)
        cv.circle(image, (int(cx), int(cy)), 4, (0, 255, 255), -1, cv.LINE_AA)

        x, y, _, _ = cv.boundingRect(box)
        text_x = max(10, min(x, image.shape[1] - 190))
        text_y = max(30, min(y - 10, image.shape[0] - 10))
        label = f'Hand {index} {degree:.1f} deg'

        cv.putText(
            image,
            label,
            (text_x, text_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            label,
            (text_x, text_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (59, 255, 255),
            1,
            cv.LINE_AA,
        )

    return image


def draw_landmarks(image, hand_landmarks):
    for landmarks in hand_landmarks:
        if len(landmarks) != 21:
            continue

        for start, end in HAND_CONNECTIONS:
            start_point = tuple(landmarks[start])
            end_point = tuple(landmarks[end])
            cv.line(image, start_point, end_point, (0, 180, 0), 3, cv.LINE_AA)
            cv.line(image, start_point, end_point, (144, 238, 144), 1, cv.LINE_AA)

        for point_index, (x, y) in enumerate(landmarks):
            radius = 5 if point_index in (0, 4, 8, 12, 16, 20) else 4
            cv.circle(image, (int(x), int(y)), radius, (0, 0, 0), -1, cv.LINE_AA)
            cv.circle(image, (int(x), int(y)), radius - 1, (255, 255, 255), -1, cv.LINE_AA)

    return image


def create_models(args):
    providers = build_providers(args.providers)
    if providers is None:
        palm_detection = PalmDetection(score_threshold=args.min_detection_confidence)
        hand_landmark = HandLandmark(class_score_th=args.min_landmark_confidence)
    else:
        palm_detection = PalmDetection(
            score_threshold=args.min_detection_confidence,
            providers=providers,
        )
        hand_landmark = HandLandmark(
            class_score_th=args.min_landmark_confidence,
            providers=providers,
        )

    return palm_detection, hand_landmark


def main():
    args = get_args()
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera device: {args.device}')

    palm_detection, hand_landmark = create_models(args)
    fps_calc = CvFpsCalc(buffer_len=10)
    window_name = 'Hand Landmark Demo'

    while True:
        fps = fps_calc.get()
        key = cv.waitKey(1)
        if key == 27:
            break

        ret, image = cap.read()
        if not ret:
            break

        if not args.disable_image_flip:
            image = cv.flip(image, 1)

        hands = palm_detection(image)
        rects = hands_to_rects(image, hands)
        debug_image = image.copy()

        if len(rects) > 0:
            hand_images = rotate_and_crop_rectangle(
                image=image,
                rects_tmp=rects,
                operation_when_cropping_out_of_range='padding',
            )
            landmarks, _ = hand_landmark(hand_images, rects)
            debug_image = draw_handboxes(debug_image, rects)
            debug_image = draw_landmarks(debug_image, landmarks)

        info_text = (
            f'FPS: {fps:.1f}  Hands: {len(rects)}  '
            f'Palm: {palm_detection.providers}  Landmark: {hand_landmark.providers}'
        )
        cv.putText(
            debug_image,
            info_text,
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv.LINE_AA,
        )
        cv.putText(
            debug_image,
            info_text,
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )

        cv.imshow(window_name, debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
