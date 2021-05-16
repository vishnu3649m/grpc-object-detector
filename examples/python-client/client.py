"""
A client for demonstrating gRPC Object Detector
"""

import argparse

import grpc

from image_detection_pb2_grpc import ImageDetectionStub
from image_detection_pb2 import ImageDetectionRequest


def main(args):
    channel = grpc.insecure_channel(args.conn_string)
    obj_det_service = ImageDetectionStub(channel)

    with open(args.img_file, "rb") as img_file:
        img_bytes = img_file.read()

    request = ImageDetectionRequest(image=img_bytes)
    response = obj_det_service.DetectImage(request)

    for d in response.detections:
        print(d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Example client for gRPC Object Detection server")

    parser.add_argument("-i",
                        dest="img_file",
                        required=True,
                        help="Path to image file for detection")
    parser.add_argument("--server",
                        dest="conn_string",
                        default="127.0.0.1:8081",
                        help="Server host and port. If not provided, "
                             "'localhost:8081' will be used")

    main(parser.parse_args())
