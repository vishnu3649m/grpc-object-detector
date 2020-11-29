gRPC Video Analyzer
-------------------

A gRPC-based server that runs deep learning-based object detectors & trackers on your
videos/images to detect, count or track objects you want.

## Introduction

The main motivation for this project is to explore gRPC and to see how well it can
be used to run video analytics.

The aim is to first develop a simple server that takes in a video, processes each
frame using an object detector and responds with a stream of detections.

To test & illustrate the server, a CLI based client is provided. Maybe a web app
will be built in the future.
