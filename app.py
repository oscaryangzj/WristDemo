import argparse
from math import degrees

import cv2 as cv
import numpy as np

from model import PalmDetection, HandLandmark
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
    parser = argparse.ArgumentParser ()
    parser.add_argument ('-d', '--device', type = int, default = 0)
    parser.add_argument ('-wi', '--width', type = int, default = 640)
    parser.add_argument ('-he', '--height', type = int, default = 480)
    parser.add_argument ('-mdc', '--min_detection_confidence', type = float, default = 0.6)
    parser.add_argument ('-mlc', '--min_landmark_confidence', type = float, default = 0.5)
    parser.add_argument ('-dif', '--disable_image_flip', action = 'store_true')
    parser.add_argument (
        '--providers',
        choices = ['default', 'cpu', 'coreml'],
        default =' default',
        help = 'ONNX Runtime providers to use.',
    )
    return parser.parse_args ()


def build_providers (provider_mode):
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


def draw_palm_detections (image, hands):
    image_height, image_width = image.shape[: 2]
    wh_ratio = image_width / image_height
    for index, hand in enumerate (hands, start = 1):
        sq_size, rotation, center_x, center_y = hand

        cx = int (center_x * image_width)
        cy = int (center_y * image_height)
        rect_width = int (sq_size * image_width)
        rect_height = int (sq_size * wh_ratio * image_height)
        degree = degrees (rotation)

        rect = ((cx, cy), (rect_width, rect_height), degree)
        box = cv.boxPoints (rect).astype (np.intp)

        cv.drawContours (image, [box], 0, (0, 0, 255), 2, cv.LINE_AA)
        cv.circle (image, (cx, cy), 4, (0, 255, 255), -1, cv.LINE_AA)

        x, y, w, h = cv.boundingRect (box)
        text_x = max (10, min (x, image_width - 180))
        text_y = max (30, min (y - 10, image_height - 10))
        cv.putText (
            image,
            f'Palm {index} {degree:.1f} deg',
            (text_x, text_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv.LINE_AA,
        )
        cv.putText (
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


def hands_to_rects (image, hands):
    image_height, image_width = image.shape[: 2]
    wh_ratio = image_width / image_height
    rects = []
    for hand in hands:
        sqn_rr_size, rotation, center_x, center_y = hand
        cx = center_x * image_width
        cy = center_y * image_height
        rect_width = sqn_rr_size * image_width
        rect_height = sqn_rr_size * wh_ratio * image_height
        rects.append ( [cx, cy, rect_width, rect_height, degrees (rotation)] )
    return np.asarray (rects, dtype = np.float32)


def draw_landmarks (image, landmarks):
    for landmark in landmarks:
        if (len (landmark) != 21):
            continue

        for fr, to in HAND_CONNECTIONS:
            xfr = tuple (landmark [fr])
            xto = tuple (landmark [to])
            cv.line (image, xfr, xto, (0, 180, 0), 3, cv.LINE_AA)
            cv.line (image, xfr, xto, (144, 238, 144), 1, cv.LINE_AA)

        for index, (x, y) in enumerate (landmark):
            radius = 5 if index in (0, 4, 8, 12, 16, 20) else 4
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


def main () :
    args = get_args ()
    cap = cv.VideoCapture (args.device)
    cap.set (cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set (cv.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened ():
        raise RuntimeError (f'Could not open camera device: {args.device}')

    palm_detection, hand_landmark = create_models(args)
    fps_calc = CvFpsCalc (buffer_len = 10)
    window_name = 'Palm Detection Demo'

    while True:
        fps = fps_calc.get ()
        key = cv.waitKey (1)
        if key == 27:
            break

        ret, image = cap.read ()
        if not ret:
            break
        if not args.disable_image_flip:
            image = cv.flip (image, 1)

        hands = palm_detection (image)
        rects = hands_to_rects (image, hands)
        debug_image = image.copy ()

        if len (rects) > 0:
            hand_images = rotate_and_crop_rectangle (
                image = image,
                rects_tmp = rects,
                operation_when_cropping_out_of_range = 'padding',
            )
            landmarks, _ = hand_landmark (hand_images, rects)
            debug_image = draw_palm_detections (debug_image, hands)
            debug_image = draw_landmarks (debug_image, landmarks)

        cv.putText (
            debug_image,
            f'FPS: {fps:.1f}  Palms: {len(hands)}  Providers: {palm_detection.providers}',
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv.LINE_AA,
        )
        cv.putText (
            debug_image,
            f'FPS: {fps:.1f}  Palms: {len(hands)}  Providers: {palm_detection.providers}',
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )

        cv.imshow (window_name, debug_image)

    cap.release ()
    cv.destroyAllWindows ()


if __name__ == '__main__':
    main()






