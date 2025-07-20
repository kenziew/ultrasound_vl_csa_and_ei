# Import libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold
import torch
import tqdm
from tqdm import tqdm
import torch.nn as nn
import cv2
import os
import torch.nn.functional as F

class CFG:
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    n_folds = 5
    fold_to_run = 0
    seed = 0
    num_classes = 2
    # skin_threshold = 0.5
    subq_threshold = 0.5
    # apo_threshold = 0.5
    muscle_threshold = 0.5
    img_size = 512
    color_palette = [
            # [0, 0, 255],
            [0, 255, 0],
            # [255, 0, 0],
            [0, 255, 255],
        ]
    color_dict = {
        # "skin": [255, 0, 0],
        "subq": [0, 255, 0],        
        # "apo": [0, 0, 255],
        "muscle": [255, 255, 0]
    }
    base_path = '/media/kenz/Disc1/Ultrasound/Vastus/10262024'
    output_dir = './training_logs/gaussian_blur+colorjitter/test_set_predictions'

def dice_coef(y_true, y_pred, thr=0.5, dim=(1,2), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean()

    return dice

def iou(y_true, y_pred, num_classes, thr=0.5, dim=(1, 2), epsilon=1e-6):
    y_pred = (y_pred > thr).to(torch.float32)
    iou_per_class = []
    for cls in range(num_classes):
        true_cls = (y_true == cls).to(torch.float32)
        pred_cls = (y_pred == cls).to(torch.float32)
        intersection = (true_cls * pred_cls).sum(dim=dim)
        union = true_cls.sum(dim=dim) + pred_cls.sum(dim=dim) - intersection
        iou = (intersection + epsilon) / (union + epsilon)
        iou_per_class.append(iou.mean().item())
    return iou_per_class

def predict(df, model):
    model.eval()
    val_scores_dice = []
    val_scores_iou = []

    img_names = []
    pred_vl_areas = []
    pred_avg_grayscales = []
    pred_std_grayscales = []
    gt_vl_areas = []
    gt_avg_grayscales = []
    gt_std_grayscales = []

    pred_subq_avg_thicknesses = []
    pred_subq_std_thicknesses = []
    pred_subq_min_thicknesses = []
    pred_subq_max_thicknesses = []

    gt_subq_avg_thicknesses = []
    gt_subq_std_thicknesses = []
    gt_subq_min_thicknesses = []
    gt_subq_max_thicknesses = []

    pred_subq_thickness_first_col = []
    pred_subq_thickness_mid_col = []
    pred_subq_thickness_last_col = []

    gt_subq_thickness_first_col = []
    gt_subq_thickness_mid_col = []
    gt_subq_thickness_last_col = []
    
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        input_img = cv2.imread(os.path.join(CFG.base_path, 'exported_images', row.ImageFilename), cv2.IMREAD_GRAYSCALE)
        mask_path = os.path.join(CFG.base_path, 'exported_labels', row.MaskFilename)
        mask = cv2.imread(mask_path)
        temp = np.zeros((mask.shape[0], mask.shape[1], CFG.num_classes))

        for i, color in enumerate(CFG.color_palette):
            temp[..., i][np.where((mask==color).all(axis=2))] = 1
        mask = temp
        del temp
        mask = np.transpose(mask, (2, 0, 1))
        mask = np.expand_dims(mask, 0)
        mask = torch.tensor(mask).to(CFG.device)

        with torch.no_grad():
            img = np.array(input_img)
            img = np.expand_dims(np.expand_dims(img, 0), 0)
            img = torch.tensor(img).to(CFG.device).float() / 255.0
            img = F.interpolate(img, (CFG.img_size, CFG.img_size), mode='bilinear', align_corners=False)
            
            img = model(img)
            img = nn.Sigmoid()(img)
            img = F.interpolate(img, input_img.shape, mode='bilinear', align_corners=False)
            dice = []
            for i in range(CFG.num_classes):
                dice.append(dice_coef(mask[:, i], img[:, i]).cpu().detach().numpy()*100.0)
                val_scores_dice.append(dice)

            iou_scores = iou(mask, img, CFG.num_classes)
            val_scores_iou.append(iou_scores)
            
            img = np.array(img[0].cpu().detach().numpy())
            # img = np.transpose(img, (1, 2, 0))
            
        # skin_mask = np.zeros((input_img.shape[0], input_img.shape[1]))
        subq_mask = np.zeros((input_img.shape[0], input_img.shape[1]))
        # apo_mask = np.zeros((input_img.shape[0], input_img.shape[1]))
        muscle_mask = np.zeros((input_img.shape[0], input_img.shape[1]))

        #Order matters
        # skin_mask[img[0] >= CFG.skin_threshold] = 1
        subq_mask[img[0] >= CFG.subq_threshold] = 1
        # apo_mask[img[2] >= CFG.apo_threshold] = 1
        muscle_mask[img[1] >= CFG.muscle_threshold] = 1

        pred_vl_area = cv2.countNonZero(muscle_mask)
        pred_avg_grayscale = np.sum((input_img*muscle_mask.astype(np.float64))) / pred_vl_area
        pred_std_grayscale = np.std((input_img*muscle_mask.astype(np.float64)))

         # Calculate predicted subq tissue thickness metrics
        pred_thickness_list = []
        for col in range(input_img.shape[1]):
            if muscle_mask[:, col].any():
                first_subq_row = np.argmax(subq_mask[:, col])
                last_subq_row = input_img.shape[0] - 1 - np.argmax(subq_mask[::-1, col])
                subq_thickness = np.sum(subq_mask[first_subq_row:last_subq_row, col])
                # if subq_thickness > 0:
                pred_thickness_list.append(subq_thickness)

         # Calculate subcutaneous thickness for first, middle, and last columns
        columns_with_muscle = np.unique(np.where(muscle_mask.sum(axis=0) > 0))
        if len(columns_with_muscle) > 0:
            first_col = columns_with_muscle[0]
            mid_col = columns_with_muscle[len(columns_with_muscle) // 2]
            last_col = columns_with_muscle[-1]

            pred_subq_thickness_first_col.append(np.sum(subq_mask[:, first_col]))
            pred_subq_thickness_mid_col.append(np.sum(subq_mask[:, mid_col]))
            pred_subq_thickness_last_col.append(np.sum(subq_mask[:, last_col]))
        else:
            pred_subq_thickness_first_col.append(0)
            pred_subq_thickness_mid_col.append(0)
            pred_subq_thickness_last_col.append(0)


        pred_subq_avg_thicknesses.append(np.mean(pred_thickness_list))
        pred_subq_std_thicknesses.append(np.std(pred_thickness_list))
        pred_subq_min_thicknesses.append(np.min(pred_thickness_list))
        pred_subq_max_thicknesses.append(np.max(pred_thickness_list))

        _mask = np.zeros((input_img.shape[0], input_img.shape[1], 3))
        # _mask[skin_mask == 1] = CFG.color_dict['skin']
        _mask[subq_mask == 1] = CFG.color_dict['subq']
        # _mask[apo_mask == 1] = CFG.color_dict['apo']
        _mask[muscle_mask == 1] = CFG.color_dict['muscle']

        input_img_ = cv2.cvtColor(input_img,cv2.COLOR_GRAY2RGB).astype(np.uint8)
        overlay = cv2.addWeighted(input_img_, 0.75, _mask.astype(np.uint8), 0.25, 0)
        cv2.imwrite(f"{CFG.output_dir}/{row.ImageFilename.split('/')[-1]}", overlay)

        # skin_mask = np.zeros((input_img.shape[0], input_img.shape[1]))
        subq_mask = np.zeros((input_img.shape[0], input_img.shape[1]))
        # apo_mask = np.zeros((input_img.shape[0], input_img.shape[1]))
        muscle_mask = np.zeros((input_img.shape[0], input_img.shape[1]))

        mask = np.array(mask[0].cpu().detach().numpy())
        # skin_mask[mask[0] >= CFG.skin_threshold] = 1
        subq_mask[mask[0] >= CFG.subq_threshold] = 1
        # apo_mask[mask[2] >= CFG.apo_threshold] = 1
        muscle_mask[mask[1] >= CFG.muscle_threshold] = 1

        gt_vl_area = cv2.countNonZero(muscle_mask)
        gt_avg_grayscale = np.sum((input_img*muscle_mask.astype(np.float64))) / gt_vl_area
        gt_std_grayscale = np.std((input_img*muscle_mask.astype(np.float64)))

        gt_thickness_list = []
        for col in range(input_img.shape[1]):
            if muscle_mask[:, col].any():
                first_subq_col = np.argmax(subq_mask[:, col])
                last_subq_col = input_img.shape[0] - 1 - np.argmax(subq_mask[::-1, col])
                subq_thickness = np.sum(subq_mask[first_subq_col:last_subq_col, col])
                # if subq_thickness > 0:
                gt_thickness_list.append(subq_thickness)

        columns_with_muscle = np.unique(np.where(muscle_mask.sum(axis=0) > 0))
        if len(columns_with_muscle) > 0:
            first_col = columns_with_muscle[0]
            mid_col = columns_with_muscle[len(columns_with_muscle) // 2]
            last_col = columns_with_muscle[-1]

            gt_subq_thickness_first_col.append(np.sum(subq_mask[:, first_col]))
            gt_subq_thickness_mid_col.append(np.sum(subq_mask[:, mid_col]))
            gt_subq_thickness_last_col.append(np.sum(subq_mask[:, last_col]))
        else:
            gt_subq_thickness_first_col.append(0)
            gt_subq_thickness_mid_col.append(0)
            gt_subq_thickness_last_col.append(0)

        gt_subq_avg_thicknesses.append(np.mean(gt_thickness_list))
        gt_subq_std_thicknesses.append(np.std(gt_thickness_list))
        gt_subq_min_thicknesses.append(np.min(gt_thickness_list))
        gt_subq_max_thicknesses.append(np.max(gt_thickness_list))

        _mask = np.zeros((input_img.shape[0], input_img.shape[1], 3))
        # _mask[skin_mask == 1] = CFG.color_dict['skin']
        _mask[subq_mask == 1] = CFG.color_dict['subq']
        # _mask[apo_mask == 1] = CFG.color_dict['apo']
        _mask[muscle_mask == 1] = CFG.color_dict['muscle']

        overlay = cv2.addWeighted(input_img_, 0.75, _mask.astype(np.uint8), 0.25, 0)
        cv2.imwrite(f"{CFG.output_dir}/{row.ImageFilename.split('/')[-1].split('.')[0]}_original.tif", overlay)

        img_names.append(row.ImageFilename)
        pred_vl_areas.append(pred_vl_area)
        pred_avg_grayscales.append(pred_avg_grayscale)
        pred_std_grayscales.append(pred_std_grayscale)
        gt_vl_areas.append(gt_vl_area)
        gt_avg_grayscales.append(gt_avg_grayscale)
        gt_std_grayscales.append(gt_std_grayscale)

    val_scores_dice = np.mean(val_scores_dice, axis=0)
    mean_dice = np.mean(val_scores_dice)
    val_scores_iou = np.mean(val_scores_iou, axis=0)
    miou = np.mean(val_scores_iou)
    print(f'Test Mean Dice: {mean_dice:0.4f}')
    print(f'Test mIoU: {miou}')
    # print(f'Skin Dice Coefficient: {val_scores[0]:0.4f}')
    print(f'Subcutaneous Tissue Dice Coefficient: {val_scores_dice[0]:0.4f}')
    print(f'Subcutaneous Tissue IoU: {val_scores_iou[0]:0.4f}')
    # print(f'Apeneurosis Dice Coefficient: {val_scores[2]:0.4f}')
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

    df = pd.DataFrame({
        'img_name': img_names,
        'gt_vl_area': gt_vl_areas,
        'pred_vl_area': pred_vl_areas,
        'gt_avg_grayscale': gt_avg_grayscales,
        'pred_avg_grayscale': pred_avg_grayscales,
        'gt_std_grayscale': gt_std_grayscales,
        'pred_std_grayscale': pred_std_grayscales,
        'pred_subq_avg_thickness': pred_subq_avg_thicknesses,
        'pred_subq_std_thickness': pred_subq_std_thicknesses,
        'pred_subq_min_thickness': pred_subq_min_thicknesses,
        'pred_subq_max_thickness': pred_subq_max_thicknesses,
        'gt_subq_avg_thickness': gt_subq_avg_thicknesses,
        'gt_subq_std_thickness': gt_subq_std_thicknesses,
        'gt_subq_min_thickness': gt_subq_min_thicknesses,
        'gt_subq_max_thickness': gt_subq_max_thicknesses,
        'pred_subq_thickness_first_col': pred_subq_thickness_first_col,
        'pred_subq_thickness_mid_col': pred_subq_thickness_mid_col,
        'pred_subq_thickness_last_col': pred_subq_thickness_last_col,
        'gt_subq_thickness_first_col': gt_subq_thickness_first_col,
        'gt_subq_thickness_mid_col': gt_subq_thickness_mid_col,
        'gt_subq_thickness_last_col': gt_subq_thickness_last_col,
    })

    val_scores_df.to_csv(f'{CFG.output_dir}/val_scores.csv', index=False)
    df.to_csv(f'{CFG.output_dir}/analysis.csv', index=False)
    
def main():
    os.makedirs(CFG.output_dir, exist_ok=True)
    # df = pd.read_csv('/home/kenz/Documents/BioMotion/vastus/data/5_fold_split.csv')
    # df = pd.read_csv('/home/kenz/Documents/BioMotion/vastus/data/test_data.csv')
    df = pd.read_csv("./data/test_data.csv")
    # model = torch.jit.load('/home/kenz/Documents/BioMotion/vastus_sup/models/run_3/model_trace.pt')
    model = torch.jit.load('./training_logs/gaussian_blur+colorjitter/model_trace.pt')
    model = model.to(CFG.device)
    # pred_df = df[df['fold'] == CFG.fold_to_run]
    pred_df = df
    
    predict(pred_df, model)
    
if __name__ == "__main__":
    main()
