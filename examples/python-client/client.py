"""
A client for demonstrating gRPC Object Detector
"""

import argparse
import os

import grpc

from image_detection_pb2_grpc import ImageDetectionStub
from image_detection_pb2 import ImageDetectionRequest


def _print_img_detection_result(detections):
    obj_counts = {}
    msg = []

    for det in detections:
        if det.object_name in obj_counts:
            obj_counts[det.object_name] += 1
        else:
            obj_counts[det.object_name] = 1

    for obj, count in obj_counts.items():
        msg.append(f"{obj}: {count}")

    print("\t".join(msg))


def _make_img_det_request(img_file_path):
    with open(img_file_path, "rb") as img_file:
        img_bytes = img_file.read()

    return ImageDetectionRequest(image=img_bytes)


def _generate_img_stream(folder):
    img_files = list(filter(lambda x: True if x.endswith(".jpg") else False,
                            os.listdir(folder)))
    for img in img_files:
        img_path = os.path.join(folder, img)
        print(f"sending image {img_path}")
        yield _make_img_det_request(img_path)


def main(args):
    channel = grpc.insecure_channel(args.conn_string)
    obj_det_service = ImageDetectionStub(channel)

    if os.path.isdir(args.img):
        # if provided with a directory, use DetectMultipleImages
        responses = obj_det_service.DetectMultipleImages(_generate_img_stream(args.img))
        for response in responses:
            _print_img_detection_result(response.detections)
    else:
        # if provided with a single file, use DetectImage
        response = obj_det_service.DetectImage(_make_img_det_request(args.img))
        _print_img_detection_result(response.detections)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Example client for gRPC Object Detection server")

    parser.add_argument("-i",
                        dest="img",
                        required=True,
                        help="Path to image file (or a directory of image files) for detection")
    parser.add_argument("--server",
                        dest="conn_string",
                        default="127.0.0.1:8081",
                        help="Server host and port. If not provided, "
                             "'localhost:8081' will be used")

    main(parser.parse_args())
