"""

"""
import asyncio
import itertools
import logging
import os
import threading
import time

import grpc

from image_detection_pb2_grpc import ImageDetectionStub
from image_detection_pb2 import ImageDetectionRequest


def _log_detections(response):
    detections = response.detections
    obj_counts = {}
    msg = []

    for det in detections:
        if det.object_name in obj_counts:
            obj_counts[det.object_name] += 1
        else:
            obj_counts[det.object_name] = 1

    for obj, count in obj_counts.items():
        msg.append(f"{obj}: {count}")

    logging.info("Detections: " + "\t".join(msg))


def _make_img_det_request(img_file_path, detector):
    with open(img_file_path, "rb") as img_file:
        img_bytes = img_file.read()

    return ImageDetectionRequest(image=img_bytes, detector_name=detector)


async def detect_image_request(stub: ImageDetectionStub, req: ImageDetectionRequest):
    logging.info(f"[{threading.current_thread()}] Making request")
    response = await stub.DetectImage(req)
    logging.info(f"[{threading.current_thread()}] Request done: {_log_detections(response)}")


async def main(requests):
    async with grpc.aio.insecure_channel("127.0.0.1:8081") as channel:
        obj_det_service = ImageDetectionStub(channel)

        awaited = []
        for req in requests:
            call = asyncio.ensure_future(detect_image_request(obj_det_service,  req))
            awaited.append(call)
        await asyncio.gather(*awaited, return_exceptions=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    img_files = list(filter(lambda x: True if x.endswith(".jpg") else False,
                            os.listdir('../../tests/data')))
    img_files = list(map(lambda x: os.path.join('..', '..', 'tests', 'data', x), img_files))

    detectors = [
        # 'onnx_yolov4_coco',
        'cascade_face_detector',
        # 'random_pokemon',
    ]

    _requests = [_make_img_det_request(img, det)
                 for img, det in itertools.product(img_files, detectors)]

    start = time.perf_counter_ns()
    asyncio.get_event_loop().run_until_complete(main(_requests))
    elapsed = (time.perf_counter_ns() - start) // 1000000

    logging.info(f"Took {elapsed} msec")
