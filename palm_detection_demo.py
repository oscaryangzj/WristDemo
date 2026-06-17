#!/usr/bin/env python

import argparse
from math import degrees

import cv2 as cv
import numpy as np

from model import PalmDetection
from utils import CvFpsCalc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=int, default=0)
    parser.add_argument('-wi', '--width', type=int, default=640)
    parser.add_argument('-he', '--height', type=int, default=480)
    parser.add_argument('-mdc', '--min_detection_confidence', type=float, default=0.6)
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


def draw_palm_detections(image, hands):
    image_height, image_width = image.shape[:2]
    wh_ratio = image_width / image_height

    for index, hand in enumerate(hands, start=1):
        sqn_rr_size, rotation, center_x, center_y = hand

        cx = int(center_x * image_width)
        cy = int(center_y * image_height)
        rect_width = int(sqn_rr_size * image_width)
        rect_height = int(sqn_rr_size * wh_ratio * image_height)
        degree = degrees(rotation)

        rect = ((cx, cy), (rect_width, rect_height), degree)
        box = cv.boxPoints(rect).astype(np.intp)

        cv.drawContours(image, [box], 0, (0, 0, 255), 2, cv.LINE_AA)
        cv.circle(image, (cx, cy), 4, (0, 255, 255), -1, cv.LINE_AA)

        x, y, w, h = cv.boundingRect(box)
        text_x = max(10, min(x, image_width - 180))
        text_y = max(30, min(y - 10, image_height - 10))
        cv.putText(
            image,
            f'Palm {index} {degree:.1f} deg',
            (text_x, text_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            f'Palm {index} {degree:.1f} deg',
            (text_x, text_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (59, 255, 255),
            1,
            cv.LINE_AA,
        )

    return image


def main():
    args = get_args()
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera device: {args.device}')

    providers = build_providers(args.providers)
    if providers is None:
        palm_detection = PalmDetection(score_threshold=args.min_detection_confidence)
    else:
        palm_detection = PalmDetection(
            score_threshold=args.min_detection_confidence,
            providers=providers,
        )

    fps_calc = CvFpsCalc(buffer_len=10)
    window_name = 'Palm Detection Demo'

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
        debug_image = draw_palm_detections(image.copy(), hands)

        cv.putText(
            debug_image,
            f'FPS: {fps:.1f}  Palms: {len(hands)}  Providers: {palm_detection.providers}',
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv.LINE_AA,
        )
        cv.putText(
            debug_image,
            f'FPS: {fps:.1f}  Palms: {len(hands)}  Providers: {palm_detection.providers}',
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
