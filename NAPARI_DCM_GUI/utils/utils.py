from scipy.interpolate import UnivariateSpline
from qtpy.QtWidgets import QMessageBox
from skimage.draw import polygon
import albumentations as A
from typing import Tuple
import pandas as pd
import numpy as np
import torch
import json
import cv2
import os

# Common Data Augmentations for Evaluation
data_transforms = A.Compose([
        A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
        # A.Normalize(),
    ], p=1.0)

def load_json(path: str) -> dict:
    """Load JSON file given filepath"""
    with open(path) as f:
        param = json.load(f)

    return param

def setup_cfg(path: str) -> dict:
    """Load GUI config given filepath"""
    cfg = load_json(path)
    cfg['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg['current_index'] = 0
    cfg['cm_per_pixel'] = np.nan
    cfg['cm2_per_pixel'] = np.nan

    return cfg

def popup_message(text: str):
    """Send popup message in Napari GUI containing given text"""
    msg = QMessageBox()
    msg.setText(text)
    msg.exec_()

def confirm_action() -> bool:
    """Send confirmation message for running model(s) on all images within a folder in Napari GUI and return True if they say Yes"""
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Question)
    msg.setText("Are you sure you want to run the model on all images in the folder? This will overwrite all previous human corrections/model predictions!!")
    msg.setWindowTitle("Confirm Action")
    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    result = msg.exec_()

    return result == QMessageBox.Yes

def save_current_index(current_index: int, index_json_path: str):
    """Save current image index in Napari GUI to a JSON File for session continuity"""
    data = {"current_index": current_index}
    with open(index_json_path, 'w') as f:
        json.dump(data, f, indent=4)

def save_polygon(polygon: list, out_path: str):
    """Save editable polygon annotation to a JSON file"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([poly.tolist() for poly in polygon], f, indent=2)

def load_polygon(json_path: str) -> list:
    """Load editable polygon annotation from JSON file"""
    with open(json_path, "r") as f:
        data = json.load(f)
    return [np.array(poly) for poly in data]

def make_subdirs(input_dir: str, output_dir: str, output_mask_dir: str, img_list: list):
    """Save image paths and mirror the input folder structure for intuitive output visualization storage"""
    df = pd.DataFrame(img_list, columns=['img_path'])
    df.to_csv(os.path.join(output_dir, 'img_paths.csv'), index=False)

    for img_path in df['img_path']:
        rel_dir = os.path.relpath(os.path.dirname(img_path), start=input_dir)
        target_dir = os.path.join(output_mask_dir, rel_dir)
        os.makedirs(target_dir, exist_ok=True)
    
    return df

def mask_to_polygon(mask: np.ndarray, epsilon: float=3.0) -> list:
    """Convert a binary mask into a simplified polygon."""
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    largest = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    # Convert from (N,1,2) to (N,2), and flip (x,y)→(y,x) aka (col,row) as cv2 has them flipped
    polygon = approx[:, 0, ::-1]

    return [polygon]

def polygon_to_mask(polygons: list, shape: Tuple) -> np.ndarray:
    """Convert Napari polygons back into a binary mask of given shape."""
    mask = np.zeros(shape, dtype=np.uint8)
    for poly in polygons:
        rr, cc = polygon(poly[:, 0], poly[:, 1], shape)
        mask[rr, cc] = 1
    return mask

def make_mask(cfg: dict, input_img: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, int]:
    """Apply threshold and postprocessing on model predictions to create a clean binary mask"""
    mask = np.zeros((input_img.shape[0], input_img.shape[1]))
    mask[pred >= cfg['threshold']] = 1
    mask = mask.astype(np.uint8)

    mask, flag = postprocess_mask(mask)

    return mask, flag

def postprocess_mask(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Postprocess a binary mask by keeping only the largest continuous area"""
    flag = 0
    unprocessed_mask = mask.copy()
    try:
        contours, _ = cv2.findContours((mask).astype(np.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(contours, key=cv2.contourArea)
        mask = cv2.drawContours(mask, [c], -1, 255, -1)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        mask = (mask.astype(np.float64) / 255).astype(np.uint8)
    except:
        mask = unprocessed_mask
        flag = 1

    return mask, flag

def square_mask(shape, ratio_a: float = 0.25, ratio_b: float = 0.5):
    """Return a binary mask with a filled rectangle."""
    h, w = shape
    mask = np.zeros(shape, dtype=np.uint8)
    mask[round(h*ratio_a) : round(h*ratio_b), round(w*ratio_a) : round(w*ratio_b)] = 1
    return mask