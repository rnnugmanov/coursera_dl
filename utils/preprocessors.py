import numpy as np


def image2vector(image: np.ndarray):
    vector = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    return vector


def normalize_rows(x: np.ndarray):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    normalized_rows = np.divide(x, norm)
    return normalized_rows
