#Import libraries
import pandas as pd
import torch
import torchvision
import numpy as np
import cv2
from sklearn.model_selection import KFold
import time
import copy
import random
import math
from tqdm import tqdm
from collections import defaultdict
import segmentation_models_pytorch as smp
import multiprocessing as mp
import transformers
import glob
from colorama import Fore, Back, Style
c_ = Fore.GREEN
sr_ = Style.RESET_ALL
from torch.cuda import amp
import PIL
from PIL import Image
import gc
import os
import logging

class CFG:
    lr = 1e-3
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    epochs = 40
    warmup_epochs = 2
    n_folds = 5
    batch_size = 8
    fold_to_run = 0
    seed = 0
    base_path = "/media/kenz/Disc1/Ultrasound/Vastus/10262024"
    output_dir = "./training_logs/gaussian_blur+colorjitter"
    num_classes = 2
    bgr_color_palette = [
            # [0, 0, 255],
            [0, 255, 0],
            # [255, 0, 0],
            [0, 255, 255],
    ]
    img_size = 512

os.makedirs(CFG.output_dir, exist_ok=True)

log_file_path = os.path.join(CFG.output_dir, 'training.log')

logging.basicConfig(
    filename=log_file_path,
    filemode='a',  # Append to the log file
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)
os.makedirs(CFG.output_dir, exist_ok=True)

df = pd.read_csv("./data/5_fold_split.csv")

train_df = df[df['fold'] != CFG.fold_to_run]
val_df = df[df['fold'] == CFG.fold_to_run]

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

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

class BiomotionDataset(torch.utils.data.Dataset):

    def __init__(self, df, CFG, transforms=False, train=True):
        self.df = df
        self.transforms = transforms
        self.CFG = CFG
        self.train = train

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = os.path.join(CFG.base_path, "exported_images", row.ImageFilename)
        label_path = os.path.join(CFG.base_path, "exported_labels", row.MaskFilename)

        # Read images
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        label = cv2.imread(label_path)
        temp = np.zeros((label.shape[0], label.shape[1], self.CFG.num_classes))

        # Process label
        for i, color in enumerate(self.CFG.bgr_color_palette):
            temp[..., i][np.where((label == color).all(axis=2))] = 1
        label = temp
        del temp

        if self.transforms:
            img = cv2.resize(img, (self.CFG.img_size, self.CFG.img_size))
            label = cv2.resize(label, (self.CFG.img_size, self.CFG.img_size))

            if self.train:
                # Apply random horizontal flip
                if random.random() < 0.5:
                    img = np.flip(img, axis=1).copy()
                    label = np.flip(label, axis=1).copy()

                img = Image.fromarray(img).convert("L")  # Convert NumPy array to PIL Image in grayscale mode

                # Apply Gaussian blur using PIL
                if random.random() < 0.25:
                    r = np.random.uniform(0.1, 2.0)
                    img = img.filter(PIL.ImageFilter.GaussianBlur(radius=r))

                if random.random() < 0.25:
                    img = torchvision.transforms.ColorJitter(0.25, 0.25, 0.25, 0.125)(img)

                # cutmix_box = obtain_cutmix_box(img.size[0], p=0.25)

            # else:
                # cutmix_box = torch.zeros_like(torch.tensor(img))

            # Convert back to NumPy array and normalize
            img = np.array(img).astype(np.float32) / 255.0
            img = np.expand_dims(img, 0)  # Add channel dimension for PyTorch

            # Transpose label dimensions to match PyTorch format
            label = np.transpose(label, (2, 0, 1))

        return torch.tensor(img), torch.tensor(label)


train_dataset = BiomotionDataset(train_df, CFG, transforms=True, train=True)
val_dataset = BiomotionDataset(val_df, CFG, transforms=True, train=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=mp.cpu_count(), shuffle=False)

loss = smp.losses.SoftBCEWithLogitsLoss()
loss = loss.to(CFG.device)

model = smp.Unet(
    encoder_name="resnet50",
    classes = CFG.num_classes,
    in_channels=1,
)
model = model.to(device=CFG.device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_steps = len(train_loader)
scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=CFG.warmup_epochs*num_steps, num_training_steps=(CFG.epochs-CFG.warmup_epochs)*num_steps)

def dice_coef(y_true, y_pred, thr=0.5, dim=(1,2), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean()

    return dice

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion_l):
    model.train()
    scaler = amp.GradScaler()

    running_loss = 0.0
    dataset_size = 0
    # pbar = tqdm(dataloader, total=len(dataloader), desc=' Train ')

    for i, (img_x, mask_x) in enumerate(dataloader):
        img_x, mask_x = img_x.to(CFG.device), mask_x.to(CFG.device).to(torch.float32)

        # img_x[cutmix_box.unsqueeze(1).expand(img_x.shape)] = img_x.flip(0)[cutmix_box.unsqueeze(1).expand(img_x.shape)]
        # mask_x[cutmix_box.unsqueeze(1).expand(mask_x.shape)] = mask_x.flip(0)[cutmix_box.unsqueeze(1).expand(mask_x.shape)]

        with amp.autocast(enabled=True):
            model.train()
            pred_x = model(img_x)
            loss = criterion_l(pred_x, mask_x)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        
        running_loss += (loss.item() * img_x.size(0))
        dataset_size += img_x.size(0)

        epoch_loss = running_loss / dataset_size

        current_lr = optimizer.param_groups[0]['lr']

        if i % 5 == 0:
            logging.info(f'Iter: {i}/{len(dataloader)}, Train Loss: {epoch_loss:0.4f}, LR: {current_lr:0.5f}')
        # pbar.set_postfix(train_loss = f'{epoch_loss:0.4f}',
                        # lr = f'{current_lr:0.5f}')
        
    torch.cuda.empty_cache()
    gc.collect()

def validate_one_epoch(model, dataloader, criterion):
    model.eval()

    running_loss = 0.0
    dataset_size = 0

    val_scores_dice = []
    val_scores_iou = []

    pbar = tqdm(dataloader, total=len(dataloader), desc=' Valid')
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(CFG.device)
            masks = masks.to(CFG.device)

            pred = model(images)
            masks = masks.to(torch.float32)
            loss = criterion(pred, masks)

            running_loss += (loss.item() * images.size(0))
            dataset_size += images.size(0)

            epoch_loss = running_loss / dataset_size

            pred = torch.nn.Sigmoid()(pred)
            dice = []
            for i in range(CFG.num_classes):
                dice.append(dice_coef(masks[:, i], pred[:, i]).cpu().detach().numpy()*100.0)

            val_scores_dice.append(dice)
            val_scores_iou.append(iou(masks, pred, CFG.num_classes))

            pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}')

        val_scores_dice = np.mean(val_scores_dice, axis=0)
        val_scores_iou = np.mean(val_scores_iou, axis=0)
        torch.cuda.empty_cache()
        gc.collect()

    return epoch_loss, val_scores_dice, val_scores_iou

def run_training(model, optimizer, scheduler, loss):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = -np.inf
    best_epoch = -1
    history = defaultdict(list)

    for epoch in range(1, CFG.epochs+1):
        gc.collect()
        logging.info(f'Epoch {epoch}/{CFG.epochs}')

        train_loss = train_one_epoch(model, dataloader=train_loader, optimizer=optimizer, scheduler=scheduler, criterion_l=loss)
        val_loss, val_scores_dice, val_scores_iou = validate_one_epoch(model, dataloader=val_loader, criterion=loss)
        mean_dice = np.mean(val_scores_dice)
        miou = np.mean(val_scores_iou)

        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_scores_dice)
        history['Valid mIoU'].append(val_scores_iou)

        logging.info(f'Valid Mean Dice: {mean_dice:0.4f}')
        logging.info(f'Valid mIoU: {miou:0.4f}')
        # logging.info(f'Skin Dice Coefficient: {val_scores[0]:0.4f}')
        logging.info(f'Subcutaneous Tissue Dice Coefficient: {val_scores_dice[0]:0.4f}')
        logging.info(f'Subcutaneous Tissue IoU: {val_scores_iou[0]:0.4f}')
        # logging.info(f'Apeneurosis Dice Coefficient: {val_scores[2]:0.4f}')
        logging.info(f'Vastus Lateralis Dice Coefficient: {val_scores_dice[1]:0.4f}')
        logging.info(f'Vastus Lateralis IoU: {val_scores_iou[1]:0.4f}')

        if mean_dice >= best_dice:
            logging.info(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {mean_dice:0.4f})")
            best_dice = mean_dice
            best_miou = miou
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"{CFG.output_dir}/best_epoch-{CFG.fold_to_run:02d}.bin"
            torch.save(model.state_dict(), PATH)
            logging.info(f"Model Saved{sr_}")
        
        last_model_wts = copy.deepcopy(model.state_dict())
        PATH = f"{CFG.output_dir}/last_epoch-{CFG.fold_to_run:02d}.bin"
        torch.save(model.state_dict(), PATH)


    end = time.time()
    time_elapsed = end-start
    logging.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    logging.info(f'Best Score: {best_dice:0.4f}')
    logging.info(f'Best mIoU: {best_miou:0.4f}')

    model.load_state_dict(best_model_wts)

    return model, history

model, history = run_training(model, optimizer, scheduler, loss)
trace = torch.jit.trace(model, torch.randn((1, 1, 512, 512)).to(CFG.device))
torch.jit.save(trace, f'{CFG.output_dir}/model_trace.pt')
