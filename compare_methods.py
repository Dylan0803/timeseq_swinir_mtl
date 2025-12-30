# -*- coding: utf-8 -*-
# compare_methods.py
# 统一对比：LR、HR、Bicubic(Yb)、GKDM(Yk)、SwinIR_GDM(Ys)、SwinIR_Multi(Ym)
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from argparse import Namespace

# 数据与现有模块
from datasets.h5_dataset import MultiTaskDataset, generate_train_valid_test_dataset
from models.network_swinir_multi_gdm import SwinIRMulti as SwinIRMultiGDM
from models.network_swinir_multi import SwinIRMulti
from models.network_swinir_multi_enhanced import SwinIRMultiEnhanced
from models.network_swinir_hybrid import SwinIRHybrid
from models.network_swinir_fuse import SwinIRFuse

# 复用 KDM 与 Bicubic 中的实现
from bicubic import AdvancedBicubicInterpolation
from eval_gkdm import load_data_from_swinir_h5, get_gaussian_kdm_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description='统一对比：Bicubic/KDM/SwinIR(GDM & Multi)')
    # 通用
    parser.add_argument('--data_path', type=str, required=True, help='H5 数据路径')
    parser.add_argument('--save_dir', type=str,
                        default='compare_outputs', help='结果保存目录')
    parser.add_argument('--device', type=str,
                        default='cuda', help='设备：cuda 或 cpu')
    parser.add_argument('--upsampler', type=str, default='nearest+conv',
                        choices=['nearest+conv', 'pixelshuffle'], help='上采样方式（模型）')
    parser.add_argument('--config', type=str, default=None,
                        help='可选：training_args.json，复用模型结构参数')
    parser.add_argument('--scale_factor', type=int,
                        default=6, help='上采样倍率 / 下采样倍率，一般=6')

    # 样本选择逻辑（与 evaluate_multi / bicubic / eval_gkdm 对齐）
    parser.add_argument('--test_mode', type=str, default='generalization',
                        choices=['generalization', 'test_set',
                                 'all_generalization', 'all_test_set'],
                        help='测试模式')
    parser.add_argument('--sample_specs', type=str, default=None,
                        help='泛化样本规格：wind1_0,s1,50;wind2_0,s2,30')
    parser.add_argument('--test_indices', type=str, default=None,
                        help='测试集索引：1,2,3')
    parser.add_argument('--pick', type=int, default=0,
                        help='当匹配到多个样本时，取第几个（0-based），默认取第一个')

    # 两个神经网络权重
    parser.add_argument('--model_path_gdm', type=str,
                        required=True, help='SwinIR_GDM 权重路径')
    parser.add_argument('--model_path_multi', type=str,
                        required=True, help='SwinIR_Multi 权重路径')
    # SwinIR_Multi 的变体选择（保持与 evaluate_multi 一致）
    parser.add_argument('--multi_variant', type=str, default='original',
                        choices=['original', 'enhanced', 'hybrid', 'fuse'],
                        help='SwinIR_Multi 变体')

    # KDM 物理参数（与 eval_gkdm 对齐，Rco 自动给一个经验值，也可按需扩展成网格搜索）
    parser.add_argument('--mat_width', type=float, default=9.6, help='物理宽度(m)')
    parser.add_argument('--mat_height', type=float,
                        default=9.6, help='物理高度(m)')
    parser.add_argument('--kdm_rco', type=float,
                        default=1.0, help='KDM 的 Rco，Gama=Rco/3')

    # Bicubic 参数（与 bicubic.py 对齐的关键项）
    parser.add_argument('--bicubic_method', type=str, default='cubic',
                        choices=['linear', 'nearest', 'cubic', 'quintic'], help='插值方法')
    parser.add_argument('--bicubic_smooth_sigma', type=float,
                        default=0.0, help='高斯平滑 sigma')
    parser.add_argument('--bicubic_edge_enhance',
                        action='store_true', help='边缘增强')
    parser.add_argument('--bicubic_edge_strength',
                        type=float, default=0.1, help='边缘增强强度')
    parser.add_argument('--bicubic_savgol_window', type=int,
                        default=None, help='Savitzky-Golay 窗口')

    return parser.parse_args()


def select_indices_by_mode(args):
    dataset = None
    indices = []
    use_batch = False

    if args.test_mode == 'generalization':
        if args.sample_specs is None:
            raise ValueError('generalization 模式需要 --sample_specs')
        dataset = MultiTaskDataset(args.data_path, shuffle=False)
        # 与 evaluate_multi.get_dataset_indices 一致的查找（严格按 specs 顺序，存在才加入）
        index_map = {}
        for i in range(len(dataset)):
            try:
                info = dataset.data_indices[i]
                key = f"{info['wind_group']},{info['source_group']},{info['time_step']}"
                index_map[key] = i
            except Exception:
                continue
        specs = [s.strip() for s in args.sample_specs.split(';') if s.strip()]
        for s in specs:
            idx = index_map.get(s)
            if idx is not None:
                indices.append(idx)
        if not indices:
            raise ValueError('未找到匹配样本，请检查 sample_specs')
    elif args.test_mode == 'all_generalization':
        dataset = MultiTaskDataset(args.data_path, shuffle=False)
        indices = list(range(len(dataset)))
        use_batch = True
    elif args.test_mode == 'all_test_set':
        train_dataset, valid_dataset, test_dataset = generate_train_valid_test_dataset(
            args.data_path, seed=42)
        dataset = test_dataset
        indices = list(range(len(dataset)))
        use_batch = True
    else:  # test_set
        # 统一从测试集划分中取数据，并严格使用测试集内的索引（通常范围 0..1499）
        train_dataset, valid_dataset, test_dataset = generate_train_valid_test_dataset(
            args.data_path, seed=42)
        dataset = test_dataset
        if args.test_indices is None:
            raise ValueError('test_set 模式需要 --test_indices')

        raw = [int(v.strip())
               for v in args.test_indices.split(',') if v.strip()]
        max_idx = len(dataset) - 1
        for v in raw:
            if not (0 <= v <= max_idx):
                raise ValueError(f'测试集索引越界: {v}，应在 0–{max_idx}')
        indices = raw

    return dataset, indices, use_batch


def create_models(args):
    # 复用 evaluate_multi.py 的默认参数
    base_params = {
        'img_size': 16,
        'in_chans': 1,
        'upscale': args.scale_factor,
        'img_range': 1.,
        'upsampler': args.upsampler
    }
    original_params = {
        **base_params,
        'window_size': 8,
        'depths': [6, 6, 6, 6],
        'embed_dim': 60,
        'num_heads': [6, 6, 6, 6],
        'mlp_ratio': 2.,
    }
    enhanced_params = {
        **base_params,
        'window_size': 8,
        'depths': [6, 6, 6, 6],
        'embed_dim': 60,
        'num_heads': [6, 6, 6, 6],
        'mlp_ratio': 2.,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.1,
        'norm_layer': nn.LayerNorm,
        'ape': False,
        'patch_norm': True,
        'use_checkpoint': False,
        'resi_connection': '1conv'
    }
    hybrid_params = dict(enhanced_params)
    fuse_params = {
        'img_size': 16,
        'in_chans': 1,
        'upscale': args.scale_factor,
        'img_range': 1.,
        'upsampler': args.upsampler,
        'window_size': 8,
        'depths': [6, 6, 6, 6],
        'embed_dim': 60,
        'num_heads': [6, 6, 6, 6],
        'mlp_ratio': 2.,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.1,
        'norm_layer': nn.LayerNorm,
        'ape': False,
        'patch_norm': True,
        'use_checkpoint': False,
    }

    # 可选：从 config 覆盖模型结构
    def build_params(base, override_cfg):
        if override_cfg is None:
            return base
        ignore_keys = {
            'model_type', 'model_path', 'data_path', 'save_dir', 'device', 'num_samples',
            'test_mode', 'sample_specs', 'test_indices', 'seed', 'batch_size', 'num_epochs',
            'lr', 'weight_decay', 'use_test_set', 'upsampler'
        }
        p = dict(base)
        for k, v in override_cfg.items():
            if k not in ignore_keys:
                p[k] = v
        return p

    cfg = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = json.load(f)

    gdm_params = build_params(original_params, cfg)
    multi_map = {
        'original': original_params,
        'enhanced': enhanced_params,
        'hybrid': hybrid_params,
        'fuse': fuse_params
    }
    multi_params = build_params(multi_map[args.multi_variant], cfg)

    model_gdm = SwinIRMultiGDM(**gdm_params)
    if args.multi_variant == 'original':
        model_multi = SwinIRMulti(**multi_params)
    elif args.multi_variant == 'enhanced':
        model_multi = SwinIRMultiEnhanced(**multi_params)
    elif args.multi_variant == 'hybrid':
        model_multi = SwinIRHybrid(**multi_params)
    else:
        model_multi = SwinIRFuse(**multi_params)

    return model_gdm, model_multi


def load_weights(model, ckpt_path):
    torch.serialization.add_safe_globals([Namespace])
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)


def tensor_to_np(img):
    return img.squeeze().detach().cpu().numpy()


def save_img(np_img, save_path, title=None, cmap='viridis'):
    plt.figure(figsize=(5, 5))
    plt.imshow(np_img, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_combined_comparison(lr_np, hr_np, yb_np, yk_np, ys_np, ym_np, save_path,
                             bicubic_psnr=None, kdm_psnr=None, gdm_psnr=None, multi_psnr=None):
    """
    生成6张图片的横向组合对比图
    顺序：LR、HR、双三次插值、KDM、SwinIR_GDM、SwinIR_Multi
    """
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))

    # 设置图片标题
    titles = ['LR', 'HR (Ground Truth)', 'Bicubic', 'KDM',
              'SwinIR_GDM', 'SwinIR_Multi']

    # 准备数据
    images = [lr_np, hr_np, yb_np, yk_np, ys_np, ym_np]
    psnr_values = [None, None, bicubic_psnr, kdm_psnr, gdm_psnr, multi_psnr]

    for i, (img, title, psnr) in enumerate(zip(images, titles, psnr_values)):
        if img is not None:
            im = axes[i].imshow(img, cmap='viridis')
            if psnr is not None:
                title_with_psnr = f'{title}\nPSNR: {psnr:.2f} dB'
            else:
                title_with_psnr = title
            axes[i].set_title(title_with_psnr, fontsize=12)
        else:
            axes[i].text(0.5, 0.5, 'N/A', ha='center',
                         va='center', transform=axes[i].transAxes)
            axes[i].set_title(title, fontsize=12)

        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"组合对比图已保存: {save_path}")


def build_lr_index_from_hr(hr_np2d, scale_factor):
    scale_factor = int(np.floor(scale_factor))
    h, w = hr_np2d.shape
    out_h = h // scale_factor
    out_w = w // scale_factor
    res_h = h - out_h * scale_factor
    res_w = w - out_w * scale_factor
    start_h = res_h // 2
    start_w = res_w // 2
    lr_index_mat = np.zeros((out_h, out_w, 2), dtype=np.int64)
    for i in range(out_h):
        for j in range(out_w):
            hr_x = start_h + i * scale_factor
            hr_y = start_w + j * scale_factor
            lr_index_mat[i, j, 0] = hr_x
            lr_index_mat[i, j, 1] = hr_y
    return lr_index_mat


def main():
    args = parse_args()

    # 设置保存目录为脚本同目录下的 compare_outputs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_root = os.path.join(script_dir, args.save_dir)
    os.makedirs(save_root, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 选择样本
    dataset, indices, _ = select_indices_by_mode(args)
    if not indices:
        raise ValueError('没有可用样本')

    print(f"将评估 {len(indices)} 个样本: {indices}")

    # 循环处理每个样本
    for sample_idx, pick_idx in enumerate(indices):
        print(f"\n=== 处理样本 {sample_idx + 1}/{len(indices)}: 索引 {pick_idx} ===")

        sample = dataset[pick_idx]
        lr = sample['lr'].unsqueeze(0).to(device)   # [1,1,H,W]
        hr = sample['hr'].unsqueeze(0).to(device)   # [1,1,H*,W*]
        info = dataset.data_indices[pick_idx]
        wind_group = info['wind_group']
        source_group = info['source_group']
        time_step = info['time_step']

        # 1) 保存 LR/HR 原图
        lr_np = tensor_to_np(lr)
        hr_np = tensor_to_np(hr)
        save_img(lr_np, os.path.join(save_root, f's{pick_idx}_LR.png'), 'LR')
        save_img(hr_np, os.path.join(save_root, f's{pick_idx}_HR.png'), 'HR')

        # 2) Bicubic(Yb)：使用 AdvancedBicubicInterpolation
        bicubic = AdvancedBicubicInterpolation(
            upscale_factor=args.scale_factor,
            method=args.bicubic_method,
            smooth_sigma=args.bicubic_smooth_sigma,
            edge_enhance=args.bicubic_edge_enhance,
            edge_strength=args.bicubic_edge_strength,
            savgol_window=args.bicubic_savgol_window,
        )
        yb = bicubic.interpolate(lr)  # [1,1,H*,W*]
        yb_np = tensor_to_np(yb)
        save_img(yb_np, os.path.join(
            save_root, f's{pick_idx}_Yb_bicubic.png'), 'Bicubic')

        # 3) GKDM(Yk)：复用当前 test_dataset 的同一条样本 lr/hr
        hr_np2d = hr.squeeze().detach().cpu().numpy()
        lr_np2d = lr.squeeze().detach().cpu().numpy()
        lr_index_mat = build_lr_index_from_hr(hr_np2d, args.scale_factor)

        sparse_mat = np.zeros_like(hr_np2d)
        H, W = lr_np2d.shape
        for i in range(H):
            for j in range(W):
                x, y = lr_index_mat[i, j].astype(int)
                sparse_mat[x, y] = lr_np2d[i, j]

        rco = args.kdm_rco
        gama = rco / 3.0
        yk_np = get_gaussian_kdm_matrix(
            measure_mat=sparse_mat,
            mat_real_size=(args.mat_width, args.mat_height),
            sample_index_mat=lr_index_mat,
            sample_conc_mat=lr_np2d,
            Rco=rco,
            Gama=gama
        )
        yk_np = np.clip(yk_np, hr_np2d.min(), hr_np2d.max())
        save_img(yk_np, os.path.join(
            save_root, f's{pick_idx}_Yk_kdm.png'), f'KDM (Rco={rco})')

        # 4) SwinIR_GDM(Ys)
        model_gdm, model_multi = create_models(args)
        load_weights(model_gdm, args.model_path_gdm)
        model_gdm = model_gdm.to(device).eval()
        with torch.no_grad():
            ys = model_gdm(lr)  # [1,1,H*,W*]
        ys_np = tensor_to_np(ys)
        save_img(ys_np, os.path.join(
            save_root, f's{pick_idx}_Ys_swinir_gdm.png'), 'SwinIR_GDM')

        # 5) SwinIR_Multi(Ym)
        load_weights(model_multi, args.model_path_multi)
        model_multi = model_multi.to(device).eval()
        with torch.no_grad():
            ym, _ = model_multi(lr)  # [1,1,H*,W*], [1,2] (gsl)
        ym_np = tensor_to_np(ym)
        save_img(ym_np, os.path.join(save_root, f's{pick_idx}_Ym_swinir_multi_{args.multi_variant}.png'),
                 f'SwinIR_Multi({args.multi_variant})')

        # 计算PSNR指标
        def psnr_np(a, b):
            mse = np.mean((a - b) ** 2)
            return 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')

        print(f'[Sample idx={pick_idx}]')
        print(f'  Bicubic  PSNR: {psnr_np(yb_np, hr_np):.2f} dB')
        print(f'  KDM      PSNR: {psnr_np(yk_np, hr_np):.2f} dB')
        print(f'  SwinIR_GDM  PSNR: {psnr_np(ys_np, hr_np):.2f} dB')
        print(f'  SwinIR_Multi PSNR: {psnr_np(ym_np, hr_np):.2f} dB')

        # 生成6张图片的横向组合对比图
        combined_save_path = os.path.join(
            save_root, f's{pick_idx}_combined_comparison.png')
        save_combined_comparison(
            lr_np=lr_np,
            hr_np=hr_np,
            yb_np=yb_np,
            yk_np=yk_np,
            ys_np=ys_np,
            ym_np=ym_np,
            save_path=combined_save_path,
            bicubic_psnr=psnr_np(yb_np, hr_np),
            kdm_psnr=psnr_np(yk_np, hr_np),
            gdm_psnr=psnr_np(ys_np, hr_np),
            multi_psnr=psnr_np(ym_np, hr_np)
        )

    print(f'\n完成：已处理 {len(indices)} 个样本，输出到 {save_root}')


if __name__ == '__main__':
    main()
