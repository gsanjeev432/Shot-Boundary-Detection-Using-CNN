import numpy as np
from PIL import Image, ImageDraw


def scenes_from_predictions(predictions: np.ndarray, threshold: float = 0.1):
    predictions = (predictions > threshold).astype(np.uint8)

    scenes = []
    t, tp, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if tp == 1 and t == 0:
            start = i
        if tp == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        tp = t
    if t == 0:
        scenes.append([start, i])
    return np.array(scenes, dtype=np.int32)
