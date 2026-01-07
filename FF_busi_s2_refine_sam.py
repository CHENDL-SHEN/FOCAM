# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import os.path as osp
from pathlib import Path
from tqdm import tqdm
import shutil
import copy
import argparse
import traceback

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F

# SAM
from segment_anything import sam_model_registry
from segment_anything.predictor import SamPredictor


# ===================== base utils =====================
def create_directory(path):
    if not osp.isdir(path):
        os.makedirs(path)
    return path

def read_image_bgr_3(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"fail to read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def ensure_bgr3(arr: np.ndarray) -> np.ndarray:
    if arr is None:
        raise ValueError("ensure_bgr3 got None")
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        return arr
    raise ValueError(f"Unexpected array ndim={arr.ndim}")

def stack_h(imgs, pad=10, bg=(255, 255, 255)):
    imgs = [ensure_bgr3(x) for x in imgs]
    H = max(im.shape[0] for im in imgs)
    outs = []
    for im in imgs:
        if im.shape[0] < H:
            top = H - im.shape[0]
            im = cv2.copyMakeBorder(im, 0, top, 0, 0, cv2.BORDER_CONSTANT, value=bg)
        outs.append(im)
    total_w = sum(im.shape[1] for im in outs) + pad * (len(outs) - 1)
    out = np.full((H, total_w, 3), bg, dtype=np.uint8)
    x = 0
    for i, im in enumerate(outs):
        h, w = im.shape[:2]
        out[:h, x:x+w] = im
        x += w
        if i < len(outs) - 1:
            x += pad
    return out

def get_hwcord(mask_path):
    mask = Image.open(mask_path)
    arr = np.asarray(mask)
    rows = np.any(arr, axis=1)
    cols = np.any(arr, axis=0)
    h_idx = np.where(rows)[0]
    w_idx = np.where(cols)[0]
    if h_idx.size == 0 or w_idx.size == 0:
        H, W = arr.shape[:2]
        return 0, H - 1, 0, W - 1
    hmin, hmax = h_idx[[0, -1]]
    wmin, wmax = w_idx[[0, -1]]
    return hmin, hmax, wmin, wmax

def colorize_prob(prob_01):
    prob_8u = np.clip(prob_01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(prob_8u, cv2.COLORMAP_JET)

def resize_keep_h(img, target_h=384):
    img = ensure_bgr3(img)
    h, w = img.shape[:2]
    new_w = max(1, round(w * target_h / h))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)


# ===================== safe upsample =====================
def safe_bilinear_resize_2d(x: torch.Tensor, size_hw):
    """
    把任意 2D/3D/4D 图像张量安全地插值到 size_hw=(H,W)。
    - 2D (H,W)     -> (1,1,H,W) 后插值 -> 回 2D
    - 3D (C,H,W)   -> (1,C,H,W) 后插值 -> 回 3D
    - 4D (N,C,H,W) -> 直接插值          -> 回 4D
    """
    if x.ndim == 2:
        x4 = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        y4 = F.interpolate(x4, size=size_hw, mode='bilinear', align_corners=False)
        return y4[0, 0]
    elif x.ndim == 3:
        x4 = x.unsqueeze(0)  # (1,C,H,W)
        y4 = F.interpolate(x4, size=size_hw, mode='bilinear', align_corners=False)
        return y4[0]
    elif x.ndim == 4:
        return F.interpolate(x, size=size_hw, mode='bilinear', align_corners=False)
    else:
        raise ValueError(f"safe_bilinear_resize_2d: unexpected ndim={x.ndim}")


# ===================== SAM logits align =====================
def logits_to_prob_aligned(lowres_logits_np, predictor):
    """
    将 SAM 的 lowres_logits 规整并经过 postprocess 对齐到原图，返回 (H,W) 概率图。
    兼容输入维度：(256,256) / (1,256,256) / (N,256,256) / (N,1,256,256)
    """
    logits = np.asarray(lowres_logits_np)

    # 统一成 (N,256,256)
    if logits.ndim == 2:  # (256,256)
        logits = logits[None, ...]
    elif logits.ndim == 4 and logits.shape[1] == 1:  # (N,1,256,256) -> (N,256,256)
        logits = logits[:, 0, :, :]
    elif logits.ndim != 3:
        logits = np.squeeze(logits)
        if logits.ndim == 2:
            logits = logits[None, ...]
        elif logits.ndim != 3:
            raise ValueError(f"Unexpected lowres_logits shape: {lowres_logits_np.shape}")

    logits = logits[:1, ...]  # 取第0张
    t = torch.from_numpy(logits).to(torch.float32).unsqueeze(1)  # (1,1,256,256) on CPU

    prob = torch.sigmoid(
        predictor.model.postprocess_masks(
            t.to(predictor.device),
            input_size=predictor.input_size,
            original_size=predictor.original_size
        )
    )[0, 0]  # (H,W)
    return prob.detach().cpu().numpy()


# ===================== sample prompt from CAM =====================
def sample_points_from_cam(cam_01: np.ndarray,
                           k: int = 5,
                           min_dist: int = 16,
                           thr: float = 0.3) -> np.ndarray:
    """
    cam_01: (H,W) in [0,1]
    返回 points: (P,2) (x,y) float32
    """
    H, W = cam_01.shape
    cam = cam_01.copy()
    cam[cam < thr] = 0.0
    points = []

    if min_dist > 0:
        yx = np.ogrid[-min_dist:min_dist + 1, -min_dist:min_dist + 1]
        mask = yx[0] ** 2 + yx[1] ** 2 <= min_dist ** 2
        kernel = mask.astype(np.uint8)
        kr, kc = kernel.shape[0] // 2, kernel.shape[1] // 2
    else:
        kernel = None

    for _ in range(k):
        idx = np.argmax(cam)
        vmax = cam.flat[idx]
        if vmax <= 0:
            break
        y, x = np.unravel_index(idx, cam.shape)
        points.append((float(x), float(y)))
        if kernel is not None:
            y0, y1 = max(0, y - kr), min(H, y + kr + 1)
            x0, x1 = max(0, x - kc), min(W, x + kc + 1)
            ky0 = kr - (y - y0)
            kx0 = kc - (x - x0)
            ky1 = ky0 + (y1 - y0)
            kx1 = kx0 + (x1 - x0)
            cam[y0:y1, x0:x1][kernel[ky0:ky1, kx0:kx1] > 0] = 0.0
        else:
            cam[y, x] = 0.0

    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


# ===================== using SAM  =====================
def run_sam_with_box_and_points(predictor,
                                box_xyxy,
                                points_xy=None,
                                points_lbl=None,
                                multimask_output=True):
    """
    box_xyxy: (4,) float32
    points_xy: (P,2) float32 or None
    points_lbl: (P,) int64 or None
    返回：masks, scores, lowres_logits
    """
    box_xyxy = np.asarray(box_xyxy, dtype=np.float32).reshape(4,)
    if points_xy is not None and len(points_xy) > 0:
        pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        lbl = np.asarray(points_lbl, dtype=np.int64).reshape(-1)
        return predictor.predict(
            point_coords=pts,
            point_labels=lbl,
            box=box_xyxy,
            multimask_output=multimask_output,
        )
    else:
        return predictor.predict(
            point_coords=None, point_labels=None,
            box=box_xyxy,
            multimask_output=multimask_output,
        )


# ===================== main process =====================
def get_refine_pselabel_from_sam_with_cam_box_point(
    img_path,
    pselabel_path,           # pselabel png:0/1/2 
    cam_npy_dir,             # CAM .npy: (1,3,Hc,Wc)
    save_path,               # output mask: 0/255 
    domain_txt,
    sam_checkpoint="/media/ders/sdd1/XS/pipeline/weights/SAM/sam_vit_h_4b8939.pth",
    model_type="vit_h",
    device="cuda",
    multimask_output=True,
    pick_by="argmax",        # "argmax" or "fixed"
    fixed_idx=2,             # use when pick_by=fixed 
    fuse_mode="prob",        
    vis_enable=False,
    malign_box_gap=20,      
    point_k=5,
    point_min_dist=16,
    point_thr=0.3,
):
    savemask_path = create_directory(save_path)
    vis_dir = None
    if vis_enable:
        vis_dir = pselabel_path.rstrip("/") + "_sam_vis/"
        create_directory(vis_dir)

    # init SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # sample list
    img_name_list = [ln.strip() for ln in open(domain_txt).read().splitlines() if ln.strip()]
    error_log_file = osp.join(save_path.rstrip("/") + '_error_images.txt')
    with open(error_log_file, 'w') as f:
        f.write('Error images:\n')

    def pick_idx(scores_np):
        if np.ndim(scores_np) == 0:
            return 0
        return int(np.argmax(scores_np)) if pick_by == "argmax" else int(fixed_idx)

    for name in tqdm(img_name_list):
        try:
            img_file = osp.join(img_path, f"{name}.png")
            image_bgr = read_image_bgr_3(img_file)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)

            pse_file = osp.join(pselabel_path, f"{name}.png")
            if not osp.exists(pse_file):
                raise RuntimeError(f"pseudo label not found: {pse_file}")
            pse = cv2.imread(pse_file, cv2.IMREAD_UNCHANGED)
            if pse is None:
                raise RuntimeError(f"fail to read pseudo label: {pse_file}")

            hmin, hmax, wmin, wmax = get_hwcord(pse_file)
            if 'malign' in name:
                hmin += malign_box_gap; hmax -= malign_box_gap
                wmin += malign_box_gap; wmax -= malign_box_gap
            input_box = np.array([wmin, hmin, wmax, hmax], dtype=np.float32)
            masks, scores, lowres_logits = run_sam_with_box_and_points(
                predictor, input_box, None, None, multimask_output
            )
            if masks.ndim == 2:
                masks = masks[None, ...]
            H, W = masks.shape[1], masks.shape[2]
            idx0 = pick_idx(scores)
            idx0 = min(max(idx0, 0), masks.shape[0]-1)
            binary_mask0 = np.asarray(masks[idx0]).astype(bool)
            prob_np0 = logits_to_prob_aligned(lowres_logits[idx0:idx0+1, ...], predictor)  # (H,W), [0,1]

            cam_sup_np = None
            cam_path = osp.join(cam_npy_dir, f"{name}.npy")
            if osp.exists(cam_path):
                cam_arr = np.load(cam_path, allow_pickle=False)  
                if not isinstance(cam_arr, np.ndarray):
                    cam_arr = np.asarray(cam_arr)
                if cam_arr.ndim != 4 or cam_arr.shape[0] != 1 or cam_arr.shape[1] != 3:
                    raise ValueError(f"Unexpected CAM shape {cam_arr.shape} for {name}, expect (1,3,Hc,Wc)")
                cam_arr = cam_arr.astype(np.float32)
                cam_fg = cam_arr[0, 1:, ...].max(axis=0)  # (Hc,Wc)

                cmin, cmax = float(cam_fg.min()), float(cam_fg.max())
                if cmax > cmin:
                    cam_fg = (cam_fg - cmin) / (cmax - cmin + 1e-6)

                cam_sup_t = torch.from_numpy(cam_fg).to(torch.float32)  # (Hc,Wc)
                cam_sup_resized = safe_bilinear_resize_2d(cam_sup_t, size_hw=(H, W))  # (H,W)
                cam_sup_np = cam_sup_resized.cpu().numpy()

            point_coords = None
            point_labels = None
            if cam_sup_np is not None and np.any(cam_sup_np > 0):
                pts = sample_points_from_cam(cam_sup_np, k=point_k, min_dist=point_min_dist, thr=point_thr)
                if pts.shape[0] > 0:
                    point_coords = pts.astype(np.float32)            # (P,2)
                    point_labels = np.ones((point_coords.shape[0],), # (P,)
                                            dtype=np.int64)

            if point_coords is not None and point_coords.shape[0] > 0:
                masks2, scores2, lowres_logits2 = run_sam_with_box_and_points(
                    predictor, input_box, point_coords, point_labels, multimask_output
                )
                if masks2.ndim == 2:
                    masks2 = masks2[None, ...]
                idx = pick_idx(scores2)
                idx = min(max(idx, 0), masks2.shape[0]-1)
                binary_mask = np.asarray(masks2[idx]).astype(bool)
                prob_np = logits_to_prob_aligned(lowres_logits2[idx:idx+1, ...], predictor)
            else:
                binary_mask = binary_mask0
                prob_np = prob_np0

            if cam_sup_np is None or not np.any(cam_sup_np > 0):
                support = (prob_np > 0)  
            else:
                support = (cam_sup_np > 0.5)
            conf_thresh = float(prob_np[support].mean()) if np.any(support) else 0.5

            if fuse_mode == "prob":
                final = (prob_np >= conf_thresh)
            else:   # "and"
                final = np.logical_and(binary_mask, prob_np >= conf_thresh)

            out_uint8 = np.zeros((H, W), dtype=np.uint8)
            out_uint8[final] = 255
            cv2.imwrite(osp.join(savemask_path, f"{name}.png"), out_uint8)

            if vis_enable:
                tgt_h = 384
                a = resize_keep_h(image_bgr, tgt_h)
                prob_color = colorize_prob(np.clip(prob_np, 0, 1))
                b = cv2.addWeighted(resize_keep_h(image_bgr, tgt_h), 0.5,
                                    resize_keep_h(prob_color, tgt_h), 0.5, 0)
                mask_vis = np.zeros_like(ensure_bgr3(image_bgr))
                mask_vis[final] = (0, 255, 0)
                c = cv2.addWeighted(ensure_bgr3(image_bgr), 0.6, ensure_bgr3(mask_vis), 0.4, 0)
                c = resize_keep_h(c, tgt_h)
                panel = stack_h([a, b, c], pad=12)
                cv2.imwrite(osp.join(vis_dir, f"{name}_panel.jpg"), panel)

        except Exception as e:
            traceback.print_exc()
            with open(error_log_file, 'a') as f:
                f.write(name + '.png\n')
            print(f"[WARN] Error processing {name}: {e}")
        
            try:
                src = osp.join(pselabel_path, f"{name}.png")
                dst = osp.join(savemask_path, f"{name}.png")
                shutil.copy(src, dst)
            except Exception:
                pass
            continue


def convert_pselabel255_to_pselabelclassnum(pselabel255_path, save_mask_path, domain_txt):
    psesavepath = create_directory(save_mask_path)
    _ = [image_id.strip() for image_id in open(domain_txt).readlines()]  
    flag = 0
    for name in os.listdir(pselabel255_path):
        da = name[:-4]
        path = osp.join(pselabel255_path, da + '.png')
        gtboxmask_ = np.asarray(Image.open(path))
        gtboxmask = copy.deepcopy(gtboxmask_)
        shape = gtboxmask.shape
        flat = gtboxmask.reshape(-1)
        if 'benign' in da:
            flat[flat == 255] = 1
        elif 'malign' in da:
            flat[flat == 255] = 2
        gtboxmask = flat.reshape(shape)
        cv2.imwrite(osp.join(psesavepath, da + '.png'), gtboxmask)
        flag += 1
        if flag % 50 == 0:
            print(f"converted: {flag}")

if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
    IMG_DIR = "/media/ders/sdd1/XS/pipeline/dataset/BUS/Public_BUSI_v2/img/"
    PSE_DIR = "../../"
    CAM_NPY_DIR = "../../"
    DOMAIN_TXT = "../../xx.txt"
    SAVE_SAM255 = "../../"
    SAVE_SAM3 = "../../"

    get_refine_pselabel_from_sam_with_cam_box_point(
        img_path=IMG_DIR,
        pselabel_path=PSE_DIR,
        cam_npy_dir=CAM_NPY_DIR,
        save_path=SAVE_SAM255,
        domain_txt=DOMAIN_TXT,
        multimask_output=True,
        pick_by="argmax", 
        fixed_idx=2,
        fuse_mode="prob",  
        vis_enable=False,
        malign_box_gap=24,
        point_k=3,
        point_min_dist=8,
        point_thr=0.3,
    )

    convert_pselabel255_to_pselabelclassnum(
        pselabel255_path=SAVE_SAM255, save_mask_path=SAVE_SAM3, domain_txt=DOMAIN_TXT
    )

