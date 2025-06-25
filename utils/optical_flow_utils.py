import cv2
import numpy as np

def compute_optical_flow(img1, img2):
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = tvl1.calc(img1, img2, None)
    return flow

def normalize_flow(flow, bound=3.0):
    """Clip flow and normalize to 0-255"""
    flow = np.clip(flow, -bound, bound)
    flow = ((flow + bound) * (255.0 / (2 * bound))).astype(np.uint8)
    return flow

def save_flow_as_image(flow, save_path):
    """
    flow shape: (H, W, 2)
    Convert to 3 channels: x, y, magnitude
    """
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(flow_x.astype(np.float32)**2 + flow_y.astype(np.float32)**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    flow_x = cv2.normalize(flow_x, None, 0, 255, cv2.NORM_MINMAX)
    flow_y = cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX)

    flow_rgb = np.stack([flow_x, flow_y, magnitude], axis=-1).astype(np.uint8)
    cv2.imwrite(save_path, flow_rgb)
