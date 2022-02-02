import numpy as np
import cv2 as cv
import pyrealsense2 as rs
import os

pipeline = rs.pipeline()
config = rs.config()
profile = pipeline.start(config)
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
mat = np.empty((depth_frame.width, depth_frame.height))
for i in range(0, depth_frame.width):
    for j in range(0, depth_frame.height):
        mat[i][j] = depth_frame.get_distance(i, j)

print(mat)
print(depth_frame.width)
print(depth_frame.height)

os.pathsep
np.save(f"Dump\mat.npy", mat)
np.savez(f"Dump\depthFrame", depth_frame)