import statistics as stats
from typing import Tuple
import pandas as pd
from .utils import *
import numpy as np
import pydicom
import torch
import glob
import cv2
import os

def make_calc_csv(cfg: dict):
    """Create a CSV to save calculated clinical values of ultrasounds"""
    if not os.path.exists(cfg['calc_csv_path']):
        df = pd.DataFrame({
            'img_path': [],
            'review': [],
            'cm_per_pixel': [],
            'cm2_per_pixel': [],
            'vl_thickness': [],
            'vl_area': [],
            'vl_avg_echo_intensity': [],
            'vl_std_echo_intensity': [],
            'subq_thickness': [],
            'muller_adjusted_vl_avg_echo_intensity': [],
            'young_adjusted_vl_avg_echo_intensity': [],
            'corrupted_img': [],
            'conversion_fail': [],
            'vl_pred_fail': [],
            'subq_pred_fail': [],
            'calc_fail': [],
        })
        df.to_csv(cfg['calc_csv_path'], index=False)

def setup_ai_models(model_folder: str, device: str) -> list:
    """Load AI model(s) from folder"""
    model_paths = glob.glob(os.path.join(model_folder, "*.pt"))
    models = get_models(model_paths, device)
    
    return models

def load_dicom(file_path: str) -> Tuple[np.ndarray, float, float]:
    ds = pydicom.dcmread(file_path)
    image = ds.pixel_array
    try:
        region = ds.SequenceOfUltrasoundRegions[0]
        dx = abs(region.PhysicalDeltaX)
        dy = abs(region.PhysicalDeltaY)
        cm_per_pixel = dy / 10 # All measures requiring cm_per_pixel are vertical (muscle thickness)
        cm2_per_pixel = (dx / 10) * (dy / 10)
    except:
        cm_per_pixel = np.nan
        cm2_per_pixel = np.nan
    
    return image, cm_per_pixel, cm2_per_pixel


def get_mask(cfg: dict, image: np.ndarray, models: list, flags=False) -> Tuple[np.ndarray, np.ndarray]:
    """Get binary mask of predictions from model(s) with a square mask as fallback if prediction fails"""
    vl_pred, subq_pred = predict(cfg, image, models)
    vl_mask, vl_flag = make_mask(cfg, image, vl_pred)
    subq_mask, subq_flag = make_mask(cfg, image, subq_pred)

    if vl_mask.sum() < 25:
        vl_mask = square_mask(vl_mask.shape, 0.65, 0.9)
        vl_flag = 1
    if subq_mask.sum() < 25:
        subq_mask = square_mask(subq_mask.shape, 0.25, 0.5)
        subq_flag = 1

    if flags:
        return vl_mask.astype(np.uint8), subq_mask.astype(np.uint8), vl_flag, subq_flag
    return vl_mask.astype(np.uint8), subq_mask.astype(np.uint8)

def get_models(paths: list, device: str) -> list:
    """Load model(s) from list of path(s)"""
    models = []
    for model in paths:
        net = torch.jit.load(model)
        net.eval()
        net = net.to(device)
        models.append(net)

    return models

def predict(cfg: dict, input_img: np.ndarray, models: list) -> Tuple[np.ndarray, np.ndarray]:
    """Predict Vastus Lateralis and Subcutaneous Tissue masks using MyoVision-US AI model(s)"""
    ensemble_vl_pred = np.zeros((input_img.shape[0], input_img.shape[1])).astype(np.float32)
    ensemble_subq_pred = np.zeros((input_img.shape[0], input_img.shape[1])).astype(np.float32)

    img = np.array(input_img)
    img = data_transforms(image=img)['image']
    img = np.transpose(img, (2, 0, 1))
    img = img[0:1, :, :].astype(np.float32) # Turn to grayscale while keeping channel axis
    img /= 255.0
    img = np.expand_dims(img, 0)
    img = torch.tensor(img).to(cfg['device'])

    for net in models:
        pred = net(img)
        pred = torch.nn.Sigmoid()(pred)
        pred = np.array(pred[0].cpu().detach().numpy())
        pred = np.transpose(pred, (1, 2, 0))
        pred = cv2.resize(pred, (input_img.shape[1], input_img.shape[0]))
        vl_pred = pred[..., 0]
        subq_pred = pred[..., 1]
        ensemble_vl_pred += vl_pred
        ensemble_subq_pred += subq_pred

    return ensemble_vl_pred / len(models), ensemble_subq_pred / len(models)

def calculate_vl_thickness(cfg: dict, mask: np.ndarray) -> Tuple[float, int]:
    """Calculate Vastus Lateralis muscle thickness in CM"""
    fail = 0
    try:
        vl_midpoint = round(np.median(np.unique(np.argwhere(mask)[..., 1])))
        mid_length = np.sum(mask[:, vl_midpoint])
        vl_thickness = round(mid_length * cfg['cm_per_pixel'], 4)
    except:
        vl_thickness = np.nan
        fail += 1     

    return vl_thickness, fail
    
def calculate_vl_csa(cfg: dict, mask: np.ndarray) -> Tuple[float, int]:
    """Calculate Vastus Lateralis cross-sectional area in CM^2"""
    fail = 0
    try:
        num_pixels = cv2.countNonZero(mask)
        vl_area = cfg['cm2_per_pixel'] * num_pixels
        vl_area = round(vl_area, 4)
    except:
        vl_area = np.nan
        fail += 1

    return vl_area, fail
    
def calculate_vl_ei(cfg: dict, input_img: np.ndarray, mask: np.ndarray) -> Tuple[float, float, int]:
    """Calculate average and standard deviation of echo intensity of Vastus Lateralis"""
    fail = 0
    gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY).astype(np.float64)

    try:
        vl_std_ei = stats.stdev(gray[np.nonzero(mask)])
        vl_std_ei = round(vl_std_ei, 4)
    except:
        vl_std_ei = np.nan
        fail = 1

    try:
        gray *= mask.astype(np.float64)
        vl_avg_ei = np.sum(gray) / cv2.countNonZero(mask)
        vl_avg_ei = round(vl_avg_ei, 4)
    except:
        vl_avg_ei = np.nan
        fail += 2

    return vl_std_ei, vl_avg_ei, fail

def calculate_subq_thickness(cfg: dict, input_img: np.ndarray, subq_mask: np.ndarray, vl_mask: np.ndarray) -> float:
    """Calculate Subcutaneous Tissue thickness using method introduced in our paper"""
    fail = 0
    subq_thicknesses = []
    for col in range(input_img.shape[1]):
        if vl_mask[:, col].any():
            first_subq_row = np.argmax(subq_mask[:, col])
            last_subq_row = input_img.shape[0] - 1 - np.argmax(subq_mask[::-1, col])
            subq_thickness = np.sum(subq_mask[first_subq_row:last_subq_row, col])
            subq_thickness = round(subq_thickness * cfg['cm_per_pixel'], 4)
            subq_thicknesses.append(subq_thickness)

    if len(subq_thicknesses) == 0:
        fail = 1

    return np.mean(subq_thicknesses), fail

def calculate_corrected_eis(cfg: dict, vl_avg_ei: float, subq_thickness: float) -> Tuple[float, float]:
    """Calculate adjusted echo intensities using equations found by Muller et al. and Young et al."""
    muller_adjusted_vl_avg_ei = vl_avg_ei - (5.0054*(subq_thickness**2)) + (38.30836*subq_thickness)
    young_adjusted_vl_avg_ei = vl_avg_ei + (subq_thickness*40.5278)

    return round(muller_adjusted_vl_avg_ei, 4), round(young_adjusted_vl_avg_ei, 4)

def make_vl_viz(input_img: np.ndarray, vl_mask: np.ndarray, subq_mask: np.ndarray) -> np.ndarray:
    """Create visualization overlaying an ultrasound and the model predicted/human adjudicated Vastus Lateralis and Subcutaneous Tissue masks"""
    _mask = np.zeros_like(input_img)
    _mask[..., 1] = vl_mask
    _mask[..., 0] = subq_mask
    overlay = cv2.addWeighted(input_img, 0.75, np.clip(_mask, 0, 1)*255, 0.25, 0.0,)

    return overlay

def analysis(cfg: dict, images: list, models: list):
    """Run model(s) on full list of images at once"""
    muller_adjusted_vl_avg_ei_list = []
    young_adjusted_vl_avg_ei_list = []
    vl_avg_ei_list = []
    vl_std_ei_list = []
    vl_csa_list = []
    vl_thickness_list = []
    subq_thickness_list = []
    img_path_list = []
    calc_fail_list = []
    depth_fail_list = []
    cm_per_pixel_list = []
    cm2_per_pixel_list = []
    img_fail_list = []
    vl_fail_list = []
    subq_fail_list = []
    review_list = []

    for image_path in images:
        fail = 0
        calc_fail = 0
        img_flag = 0
        depth_flag = 0

        img, cm_per_pixel, cm2_per_pixel = load_dicom(image_path)
        if np.isnan(cm_per_pixel) or np.isnan(cm2_per_pixel):
            depth_flag = 1
            cm_per_pixel = 1e-3 # Default values if reading depth conversion from dicom metadata fails
            cm2_per_pixel = 1e-6
        cfg['cm_per_pixel'] = cm_per_pixel
        cfg['cm2_per_pixel'] = cm2_per_pixel

        if img is not None:
            with torch.no_grad():
                vl_mask, subq_mask, vl_flag, subq_flag = get_mask(cfg, img, models, flags=True)
                vl_poly = mask_to_polygon(vl_mask)
                subq_poly = mask_to_polygon(subq_mask)

                vl_thickness, thickness_fail = calculate_vl_thickness(cfg, vl_mask)
                vl_csa, csa_fail = calculate_vl_csa(cfg, vl_mask)
                vl_std_ei, vl_avg_ei, ei_fail = calculate_vl_ei(cfg, img, vl_mask)
                subq_thickness, subq_thickness_fail = calculate_subq_thickness(cfg, img, subq_mask, vl_mask)
                vl_viz = make_vl_viz(img, vl_mask, subq_mask)
                muller_adjusted_vl_avg_ei, young_adjusted_vl_avg_ei = calculate_corrected_eis(cfg, vl_avg_ei, subq_thickness)
                
                if thickness_fail + csa_fail + ei_fail + subq_thickness_fail > 0:
                    calc_fail = 1
                
                rel_path = os.path.relpath(image_path, start=cfg['image_folder'])
                mask_out_path = os.path.join(cfg['output_mask_path'], os.path.splitext(rel_path)[0] + "_mask.png")
                vl_poly_out_path = os.path.join(cfg['output_poly_path'], os.path.splitext(rel_path)[0] + "_vl_poly.json")
                subq_poly_out_path = os.path.join(cfg['output_poly_path'], os.path.splitext(rel_path)[0] + "_subq_poly.json")

                cv2.imwrite(str(mask_out_path), vl_viz)
                save_polygon(vl_poly, vl_poly_out_path)
                save_polygon(subq_poly, subq_poly_out_path)

        else:
            vl_thickness = np.nan
            vl_csa = np.nan
            vl_std_ei = np.nan
            vl_avg_ei = np.nan
            subq_thickness = np.nan
            muller_adjusted_vl_avg_ei = np.nan
            young_adjusted_vl_avg_ei = np.nan
            img_flag = 1

        if img_flag == 1 or vl_flag == 1 or subq_flag == 1 or depth_flag == 1 or calc_fail == 1:
            fail = 1

        vl_thickness_list.append(vl_thickness)
        vl_csa_list.append(vl_csa)
        vl_std_ei_list.append(vl_std_ei)
        vl_avg_ei_list.append(vl_avg_ei)
        subq_thickness_list.append(subq_thickness)
        muller_adjusted_vl_avg_ei_list.append(muller_adjusted_vl_avg_ei)
        young_adjusted_vl_avg_ei_list.append(young_adjusted_vl_avg_ei)
        img_path_list.append(rel_path)
        cm_per_pixel_list.append(cm_per_pixel)
        cm2_per_pixel_list.append(cm2_per_pixel)
        depth_fail_list.append(depth_flag)
        img_fail_list.append(img_flag)
        vl_fail_list.append(vl_flag)
        subq_fail_list.append(subq_flag)
        calc_fail_list.append(calc_fail)
        review_list.append(fail)

    out_df = pd.DataFrame({
        'img_path': img_path_list,
        'review': review_list,
        'cm_per_pixel': cm_per_pixel_list,
        'cm2_per_pixel': cm2_per_pixel_list,
        'vl_thickness': vl_thickness_list,
        'vl_area': vl_csa_list,
        'vl_avg_echo_intensity': vl_avg_ei_list,
        'vl_std_echo_intensity': vl_std_ei_list,
        'subq_thickness': subq_thickness_list,
        'muller_adjusted_vl_avg_echo_intensity': muller_adjusted_vl_avg_ei_list,
        'young_adjusted_vl_avg_echo_intensity': young_adjusted_vl_avg_ei_list,
        'corrupted_img': img_fail_list,
        'conversion_fail': depth_fail_list,
        'vl_pred_fail': vl_fail_list,
        'subq_pred_fail': subq_fail_list,
        'calc_fail': calc_fail_list,
    })

    out_df.to_csv(cfg['calc_csv_path'], index=False)