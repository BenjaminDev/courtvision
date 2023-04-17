import numpy as np
import cv2


def plot_coords(img:np.array, src_coords:np.array, show:bool=True):
    """Draws lines on 'img'

    Args:
        img (np.array): _description_
        src_coords (np.array): _description_
        show (bool, optional): _description_. Defaults to True.
    """
    src_coords = src_coords.astype(int)
    cv2.polylines(img, [src_coords], True, (0, 255, 0), thickness=2)
    if show:
        from matplotlib import pyplot as plt
        plt.imshow(img)