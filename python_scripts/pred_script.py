import torch.nn.functional as F
import albumentations as A
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import torch
import cv2
import os

# Common Data Augmentations for Inference
data_transforms = A.Compose([
        A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
        A.Normalize(),
    ], p=1.0)

def setup_parser():
    parser = argparse.ArgumentParser(description='VL Script')
    parser.add_argument('--models', nargs='+', type=str, default=['./vl_model.pt'],
                        help='List of one or more models to use for evaluation of vastus lateralis and subq tissue (Default: ./vl_model.pt)')
    parser.add_argument('--input-csv', type=str, default='./vl_ultrasounds.csv', 
                        help="Filepath to csv containing all filenames of ultrasounds + ground truth masks to be analyzed (Default: './vl_ultrasounds.csv')")
    parser.add_argument('--base-path', type=str, default='./vl_study', 
                        help="Filepath to folder containing all QC ultrasounds (only VI ultrasounds) within study (Default: './vl_study')")
    parser.add_argument('--output-folder-path', type=str, default="./out",
                        help="Filepath to output folder containing all analysis (Default: ./out)")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence Threshold for Predictions')

    return parser

# Retrieve models from list of paths
def get_models(args):
    models = []

    for model in args.models:
        net = torch.jit.load(model)
        net.eval()
        net = net.to(args.device)
        models.append(net)

    return models

# Calculate Dice Coefficient
def dice_coef(y_true, y_pred, thr=0.5, dim=(1,2), epsilon=1e-6):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean()

    return dice

# Calculate Intersection over Union
def iou(y_true, y_pred, thr=0.5, dim=(1,2), epsilon=1e-6):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    intersection = (y_true * y_pred).sum(dim=dim)
    union = y_true.sum(dim=dim) + y_pred.sum(dim=dim) - intersection
    iou = ((intersection + epsilon) / (union + epsilon)).mean()
    return iou

# Convert ground truth mask into tensor
def parse_mask(args, mask):
    temp = np.zeros((mask.shape[0], mask.shape[1], args.num_classes))

    for i, color in enumerate(args.color_palette):
        temp[..., i][np.where((mask==color).all(axis=2))] = 1
    mask = temp
    del temp
    mask = np.transpose(mask, (2, 0, 1))
    mask = np.expand_dims(mask, 0)
    mask = torch.tensor(mask).to(args.device)

    return mask

# Predict VL, and Subcutaneous Tissue Mask/CSA Using Models Provided
def predict(args, input_img, models):
    ensemble_vl_pred = np.zeros((input_img.shape[0], input_img.shape[1])).astype(np.float32)
    ensemble_subq_pred = np.zeros((input_img.shape[0], input_img.shape[1])).astype(np.float32)

    img = np.array(input_img)
    img = data_transforms(image=img)['image']
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.tensor(img).to(args.device)

    for net in models:
        pred = net(img)
        pred = torch.nn.Sigmoid()(pred)
        pred = np.array(pred[0].cpu().detach().numpy())
        pred = np.transpose(pred, (1, 2, 0))
        pred = cv2.resize(pred, (input_img.shape[1], input_img.shape[0]))
        vl_pred, subq_pred = pred[..., 0], pred[..., 1]
        ensemble_vl_pred += vl_pred
        ensemble_subq_pred += subq_pred

    return ensemble_vl_pred, ensemble_subq_pred

# Calculate Dice and IOU metrics for VL and Subcutaneous Tissue
def calculate_val_scores(args, vl_pred, subq_pred, mask):
    vl_dice = dice_coef(mask[:, 0], vl_pred, thr=args.threshold)*100.0
    subq_dice = dice_coef(mask[:, 1], subq_pred, thr=args.threshold)*100.0

    vl_iou = iou(mask[:, 0], vl_pred, thr=args.threshold)
    subq_iou = iou(mask[:, 1], subq_pred, thr=args.threshold)

    return [vl_dice, subq_dice], [vl_iou, subq_iou]

# Make binary numpy array from model prediction
def make_mask(args, input_img, pred):
    mask = np.zeros((input_img.shape[0], input_img.shape[1]))
    mask[pred >= args.threshold] = 1
    mask = mask.astype(np.uint8)

    return mask

# Calculate vastus lateralis cross-sectional area
def calculate_vl_csa(vl_mask):
    vl_area = cv2.countNonZero(vl_mask)

    return vl_area

# Calculate vastus lateralis echo-intensity
def calculate_vl_ei(input_img, vl_mask, vl_csa):
    avg_ei = np.sum((input_img*vl_mask.astype(np.float64))) / vl_csa
    std_ei = np.std((input_img*vl_mask.astype(np.float64)))

    return avg_ei, std_ei

# Calculate subcutaneous tissue thickness using method mentioned in our paper and return useful statistical values
def calculate_our_subq_thicknesses(input_img, subq_mask, vl_mask):
    subq_thicknesses = []
    for col in range(input_img.shape[1]):
        if vl_mask[:, col].any():
            first_subq_row = np.argmax(subq_mask[:, col])
            last_subq_row = input_img.shape[0] - 1 - np.argmax(subq_mask[::-1, col])
            subq_thickness = np.sum(subq_mask[first_subq_row:last_subq_row, col])
            subq_thicknesses.append(subq_thickness)

    return np.mean(subq_thicknesses), np.std(subq_thicknesses), np.min(subq_thicknesses), np.max(subq_thicknesses)

# Calculate subcutaneous tissue thickness using previous methods
def calculate_three_col_subq_thickness(subq_mask, vl_mask):
    cols_with_vl = np.unique(np.where(vl_mask.sum(axis=0) > 0))
    if len(cols_with_vl) > 0:
        first_col = cols_with_vl[0]
        mid_col = cols_with_vl[len(cols_with_vl) // 2]
        last_col = cols_with_vl[-1]

        return np.sum(subq_mask[:, first_col]), np.sum(subq_mask[:, mid_col]), np.sum(subq_mask[:, last_col])
    
    else:
        return 0, 0, 0
    
# Make vizualization of model predictions
def make_vizualization(input_img, subq_mask, vl_mask):
    _mask = np.zeros((input_img.shape[0], input_img.shape[1], 3))
    _mask[..., 0] = subq_mask
    _mask[..., 1] = vl_mask

    input_img_ = cv2.cvtColor(input_img,cv2.COLOR_GRAY2RGB).astype(np.uint8)
    overlay = cv2.addWeighted(input_img_, 0.75, _mask.astype(np.uint8), 0.25, 0)

    return overlay
            
# Parent function to automatically calculate vastus lateralis and subcutaneous tissue parameters given a dataframe of ultrasound images
# and ground truth masks; Returns two pandas dataframes, one of Dice and IoU metrics comparing ground truth and model predictions, and 
# another of calculated vastus lateralis and subcutaneous tissue parameters
def analysis(args, models, df):
    val_scores_dice = []
    val_scores_iou = []

    img_names = []
    vl_pred_csas = []
    avg_pred_eis = []
    std_pred_eis = []
    vl_gt_csas = []
    avg_gt_eis = []
    std_gt_eis = []

    subq_pred_avg_thicknesses = []
    subq_pred_std_thicknesses = []
    subq_pred_min_thicknesses = []
    subq_pred_max_thicknesses = []

    subq_gt_avg_thicknesses = []
    subq_gt_std_thicknesses = []
    subq_gt_min_thicknesses = []
    subq_gt_max_thicknesses = []

    subq_pred_thickness_first_col = []
    subq_pred_thickness_mid_col = []
    subq_pred_thickness_last_col = []

    subq_gt_thickness_first_col = []
    subq_gt_thickness_mid_col = []
    subq_gt_thickness_last_col = []

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        img_path = os.path.join(args.base_path, 'exported_images', row.ImageFilename)
        input_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask_path = os.path.join(args.base_path, 'exported_labels', row.MaskFilename)
        mask = cv2.imread(mask_path)

        if input_img is not None and mask_path is not None:
            with torch.no_grad():
                mask = parse_mask(args, mask)
                vl_pred, subq_pred = predict(args, input_img, models)

                dice, iou = calculate_val_scores(args, vl_pred, subq_pred, mask)
                val_scores_dice.append(dice)
                val_scores_iou.append(iou)

                vl_pred = make_mask(args, input_img, vl_pred)
                subq_pred = make_mask(args, input_img, subq_pred)
                vl_gt = make_mask(args, input_img, mask[0][0])
                subq_gt = make_mask(args, input_img, mask[0][1])

                vl_pred_csa = calculate_vl_csa(vl_pred)
                vl_gt_csa = calculate_vl_csa(vl_gt)
                vl_pred_avg_ei, vl_pred_std_ei = calculate_vl_ei(input_img, vl_pred, vl_pred_csa)
                vl_gt_avg_ei, vl_gt_std_ei = calculate_vl_ei(input_img, vl_gt, vl_gt_csa)
                vl_pred_avg_subq, vl_pred_std_subq, vl_pred_min_subq, vl_pred_max_subq = calculate_our_subq_thicknesses(input_img, subq_pred, vl_pred)
                vl_gt_avg_subq, vl_gt_std_subq, vl_gt_min_subq, vl_gt_max_subq = calculate_our_subq_thicknesses(input_img, subq_gt, vl_gt)
                vl_pred_subq_first, vl_pred_subq_mid, vl_pred_subq_last = calculate_three_col_subq_thickness(subq_pred, vl_pred)
                vl_gt_subq_first, vl_gt_subq_mid, vl_gt_subq_last = calculate_three_col_subq_thickness(subq_gt, vl_gt)
                vizualization = make_vizualization(input_img, subq_pred, vl_pred)

                cv2.imwrite(os.path.join(args.output_folder_path, row.ImageFilename.replace(args.base_path, '')), vizualization)

        else:
            raise FileNotFoundError(f"Either {img_path} OR {mask_path} was not found!")
        
        img_names.append(row.ImageFilename)
        vl_pred_csas.append(vl_pred_csa)
        vl_gt_csas.append(vl_gt_csa)
        avg_pred_eis.append(vl_pred_avg_ei)
        avg_gt_eis.append(vl_gt_avg_ei)
        std_pred_eis.append(vl_pred_std_ei)
        std_gt_eis.append(vl_gt_std_ei)

        subq_pred_avg_thicknesses.append(vl_pred_avg_subq)
        subq_gt_avg_thicknesses.append(vl_gt_avg_subq)
        subq_pred_std_thicknesses.append(vl_pred_std_subq)
        subq_gt_std_thicknesses.append(vl_gt_std_subq)
        subq_pred_min_thicknesses.append(vl_pred_min_subq)
        subq_gt_min_thicknesses.append(vl_gt_min_subq)
        subq_pred_max_thicknesses.append(vl_pred_max_subq)
        subq_gt_max_thicknesses.append(vl_gt_max_subq)

        subq_pred_thickness_first_col.append(vl_pred_subq_first)
        subq_gt_thickness_first_col.append(vl_gt_subq_first)
        subq_pred_thickness_mid_col.append(vl_pred_subq_mid)
        subq_gt_thickness_mid_col.append(vl_gt_subq_mid)
        subq_pred_thickness_last_col.append(vl_pred_subq_last)
        subq_gt_thickness_last_col.append(vl_gt_subq_last)

    val_scores_dice = np.mean(val_scores_dice, axis=0)
    mean_dice = np.mean(val_scores_dice)
    val_scores_iou = np.mean(val_scores_iou, axis=0)
    miou = np.mean(val_scores_iou)
    print(f'Test Mean Dice: {mean_dice:0.4f}')
    print(f'Test mIoU: {miou}')
    print(f'Subcutaneous Tissue Dice Coefficient: {val_scores_dice[0]:0.4f}')
    print(f'Subcutaneous Tissue IoU: {val_scores_iou[0]:0.4f}')
    print(f'Vastus Lateralis Dice Coefficient: {val_scores_dice[1]:0.4f}')
    print(f'Vastus Lateralis IoU: {val_scores_iou[1]:0.4f}')

    val_scores_df = pd.DataFrame({
        'Mean_Dice': [mean_dice],
        'mIoU': [miou],
        'SubQ Dice': [val_scores_dice[0]],
        'SubQ IoU': [val_scores_iou[0]],
        'VL Dice': [val_scores_dice[1]],
        'VL IoU': [val_scores_iou[1]],
    })

    out_df = pd.DataFrame({
        'img_name': img_names,
        'gt_vl_area': vl_gt_csas,
        'pred_vl_area': vl_pred_csas,
        'gt_avg_grayscale': avg_gt_eis,
        'pred_avg_grayscale': avg_pred_eis,
        'gt_std_grayscale': std_gt_eis,
        'pred_std_grayscale': std_pred_eis,
        'pred_subq_avg_thickness': subq_pred_avg_thicknesses,
        'pred_subq_std_thickness': subq_pred_std_thicknesses,
        'pred_subq_min_thickness': subq_pred_min_thicknesses,
        'pred_subq_max_thickness': subq_pred_max_thicknesses,
        'gt_subq_avg_thickness': subq_gt_avg_thicknesses,
        'gt_subq_std_thickness': subq_gt_std_thicknesses,
        'gt_subq_min_thickness': subq_gt_min_thicknesses,
        'gt_subq_max_thickness': subq_gt_max_thicknesses,
        'pred_subq_thickness_first_col': subq_pred_thickness_first_col,
        'pred_subq_thickness_mid_col': subq_pred_thickness_mid_col,
        'pred_subq_thickness_last_col': subq_pred_thickness_last_col,
        'gt_subq_thickness_first_col': subq_gt_thickness_first_col,
        'gt_subq_thickness_mid_col': subq_gt_thickness_mid_col,
        'gt_subq_thickness_last_col': subq_gt_thickness_last_col,
    })

    return val_scores_df, out_df
    
def main():
    args = setup_parser().parse_args()
    args.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.num_classes = 2
    args.color_palette = [
            [0, 255, 0],
            [0, 255, 255],
        ]
    os.makedirs(args.output_folder_path, exist_ok=True)
    
    df = pd.read_csv(args.input_csv)
    models = get_models(args)
    
    metrics, inference = analysis(args, models, df)

    metrics.to_csv(os.path.join(args.output_folder_path, 'val_metrics.csv'), index=False)
    inference.to_csv(os.path.join(args.output_folder_path, 'preds.csv'), index=False)
    
if __name__ == "__main__":
    main()
