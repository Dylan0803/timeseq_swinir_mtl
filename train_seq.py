import os
import sys
import time
import json
import shutil
import random
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ========= 只保留 original 多任务模型 =========
from models.network_swinir_multi import SwinIRMulti

# ========= 数据集（支持 seq）=========
from datasets.h5_dataset import generate_train_valid_test_dataset


# ------------------ utils ------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_args(args, save_dir: str):
    args_dict = vars(args)
    with open(os.path.join(save_dir, "training_args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)


def plot_loss_lines(train_losses, valid_losses,
                    train_sr_losses, valid_sr_losses,
                    train_gsl_losses, valid_gsl_losses,
                    train_pred_losses, valid_pred_losses,
                    save_dir: str,
                    enable_pred: bool):
    plt.figure(figsize=(16, 10))

    # Total
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label="Train Total")
    plt.plot(valid_losses, label="Valid Total")
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # SR@t
    plt.subplot(2, 2, 2)
    plt.plot(train_sr_losses, label="Train SR@t")
    plt.plot(valid_sr_losses, label="Valid SR@t")
    plt.title("SR@t Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # GSL
    plt.subplot(2, 2, 3)
    plt.plot(train_gsl_losses, label="Train GSL")
    plt.plot(valid_gsl_losses, label="Valid GSL")
    plt.title("GSL Loss (SmoothL1)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Pred@t+1
    plt.subplot(2, 2, 4)
    if enable_pred:
        plt.plot(train_pred_losses, label="Train Pred@t+1")
        plt.plot(valid_pred_losses, label="Valid Pred@t+1")
    plt.title("Pred@t+1 Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_metrics.png"))
    plt.close()


def save_training_history(save_dir: str,
                          train_losses, valid_losses,
                          train_sr_losses, valid_sr_losses,
                          train_gsl_losses, valid_gsl_losses,
                          train_pred_losses, valid_pred_losses,
                          enable_pred: bool):
    history = {
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_total_loss": train_losses,
        "valid_total_loss": valid_losses,
        "train_sr_loss": train_sr_losses,
        "valid_sr_loss": valid_sr_losses,
        "train_gsl_loss": train_gsl_losses,
        "valid_gsl_loss": valid_gsl_losses,
    }
    if enable_pred:
        history["train_pred_loss"] = train_pred_losses
        history["valid_pred_loss"] = valid_pred_losses

    pd.DataFrame(history).to_csv(os.path.join(
        save_dir, "training_history.csv"), index=False)


# ------------------ args ------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SwinIR Original Multi-task (Seq)")

    # 数据
    parser.add_argument("--data_path", type=str,
                        required=True, help="path to .h5 dataset")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="directory to save results")
    parser.add_argument("--use_test_set", action="store_true",
                        help="train/val/test=0.8/0.1/0.1; else 0.8/0.2")

    # 时序
    parser.add_argument("--use_seq", action="store_true",
                        help="use MultiTaskSeqDataset")
    parser.add_argument("--K", type=int, default=6,
                        help="history length K (for seq dataset)")

    # 训练
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    # loss 权重
    parser.add_argument("--sr_weight", type=float,
                        default=1.0, help="weight for SR@t")
    parser.add_argument("--gsl_weight", type=float,
                        default=0.5, help="weight for GSL")
    parser.add_argument("--enable_pred", action="store_true",
                        help="enable Pred@t+1 task")
    parser.add_argument("--pred_weight", type=float,
                        default=1.0, help="weight for Pred@t+1")

    # 上采样器
    parser.add_argument("--upsampler", type=str, default="nearest+conv",
                        choices=["nearest+conv", "pixelshuffle"])

    return parser.parse_args()


# ------------------ model ------------------
def create_model(args):
    model = SwinIRMulti(
        img_size=16,
        in_chans=1,
        upscale=6,
        img_range=1.0,
        upsampler=args.upsampler,
        window_size=8,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2.0,
    )
    return model


# ------------------ train/valid loop ------------------
def train_model(model, train_loader, valid_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    sr_criterion = nn.MSELoss()
    pred_criterion = nn.MSELoss()
    gsl_criterion = nn.SmoothL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    os.makedirs(args.save_dir, exist_ok=True)
    save_args(args, args.save_dir)

    # history
    train_losses, valid_losses = [], []
    train_sr_losses, valid_sr_losses = [], []
    train_gsl_losses, valid_gsl_losses = [], []
    train_pred_losses, valid_pred_losses = [], []

    best_valid_loss = float("inf")
    start_epoch = 0

    # resume
    checkpoint_path = os.path.join(args.save_dir, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        train_losses = ckpt["train_losses"]
        valid_losses = ckpt["valid_losses"]
        train_sr_losses = ckpt["train_sr_losses"]
        valid_sr_losses = ckpt["valid_sr_losses"]
        train_gsl_losses = ckpt["train_gsl_losses"]
        valid_gsl_losses = ckpt["valid_gsl_losses"]
        train_pred_losses = ckpt.get("train_pred_losses", [])
        valid_pred_losses = ckpt.get("valid_pred_losses", [])
        best_valid_loss = ckpt["best_valid_loss"]
        print(f"Resuming from epoch {start_epoch}")

    # ========= One-batch sanity check =========
    print("\n" + "="*60)
    print("One-batch sanity check before training...")
    print("="*60)
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        if args.use_seq:
            inp = sample_batch["lr_seq"].to(device)  # (B,K,1,16,16)
            print(
                f"✓ Input shape (seq): {inp.shape} (expected: (bs,K,1,16,16))")
        else:
            inp = sample_batch["lr"].to(device)  # (B,1,16,16)
            print(
                f"✓ Input shape (single): {inp.shape} (expected: (bs,1,16,16))")

        out = model(inp)
        sr_out, gsl_out, pred_out = out

        print(f"✓ sr_out.shape: {sr_out.shape} (expected: (bs,1,96,96))")
        print(f"✓ pred_out.shape: {pred_out.shape} (expected: (bs,1,96,96))")
        print(f"✓ gsl_out.shape: {gsl_out.shape} (expected: (bs,2))")

        assert sr_out.shape[1:] == (
            1, 96, 96), f"sr_out shape mismatch: {sr_out.shape}"
        assert pred_out.shape[1:] == (
            1, 96, 96), f"pred_out shape mismatch: {pred_out.shape}"
        assert gsl_out.shape[1:] == (
            2,), f"gsl_out shape mismatch: {gsl_out.shape}"

        print("✓ All shapes correct! Starting training...\n")
    model.train()

    for epoch in range(start_epoch, args.num_epochs):
        # -------- train --------
        model.train()
        total_loss_sum = 0.0
        sr_loss_sum = 0.0
        gsl_loss_sum = 0.0
        pred_loss_sum = 0.0

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]", ncols=150)
        for batch in pbar:
            # seq or single
            if args.use_seq:
                inp = batch["lr_seq"].to(device)          # (B,K,1,16,16)
                hr_t = batch["hr_t"].to(device)              # (B,1,96,96)
                hr_tp1 = batch["hr_tp1"].to(device)          # (B,1,96,96)
                source_pos = batch["source_pos"].to(device)  # (B,2)
            else:
                inp = batch["lr"].to(device)                 # (B,1,16,16)
                hr_t = batch["hr"].to(device)                # (B,1,96,96)
                hr_tp1 = None
                source_pos = batch["source_pos"].to(device)  # (B,2)

            optimizer.zero_grad()

            out = model(inp)
            # 模型现在返回 (sr_out, gsl_out, pred_out)
            sr_out, gsl_out, pred_out = out

            sr_loss = sr_criterion(sr_out, hr_t)
            gsl_loss = gsl_criterion(gsl_out, source_pos)

            loss = args.sr_weight * sr_loss + args.gsl_weight * gsl_loss

            # pred task (仅当 --enable_pred 开启且 hr_tp1 存在时计算)
            if args.enable_pred and hr_tp1 is not None:
                pred_loss = pred_criterion(pred_out, hr_tp1)
                loss = loss + args.pred_weight * pred_loss
                pred_loss_sum += pred_loss.item()
            else:
                pred_loss = torch.tensor(0.0, device=device)

            loss.backward()
            optimizer.step()

            total_loss_sum += loss.item()
            sr_loss_sum += sr_loss.item()
            gsl_loss_sum += gsl_loss.item()

            postfix = {
                "loss": f"{loss.item():.4f}",
                "sr": f"{sr_loss.item():.4f}",
                "gsl": f"{gsl_loss.item():.4f}",
            }
            if args.enable_pred and hr_tp1 is not None:
                postfix["pred"] = f"{pred_loss.item():.4f}"
            else:
                postfix["pred"] = "0.0"
            pbar.set_postfix(postfix)

        n_train = len(train_loader)
        train_total = total_loss_sum / n_train
        train_sr = sr_loss_sum / n_train
        train_gsl = gsl_loss_sum / n_train
        train_pred = (pred_loss_sum /
                      n_train) if (args.enable_pred and args.use_seq) else 0.0

        # -------- valid --------
        model.eval()
        total_loss_sum = 0.0
        sr_loss_sum = 0.0
        gsl_loss_sum = 0.0
        pred_loss_sum = 0.0

        pbar = tqdm(
            valid_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Valid]", ncols=150)
        with torch.no_grad():
            for batch in pbar:
                if args.use_seq:
                    inp = batch["lr_seq"].to(device)          # (B,K,1,16,16)
                    hr_t = batch["hr_t"].to(device)              # (B,1,96,96)
                    hr_tp1 = batch["hr_tp1"].to(device)          # (B,1,96,96)
                    source_pos = batch["source_pos"].to(device)  # (B,2)
                else:
                    inp = batch["lr"].to(device)                 # (B,1,16,16)
                    hr_t = batch["hr"].to(device)                # (B,1,96,96)
                    hr_tp1 = None
                    source_pos = batch["source_pos"].to(device)  # (B,2)

                out = model(inp)
                # 模型现在返回 (sr_out, gsl_out, pred_out)
                sr_out, gsl_out, pred_out = out

                sr_loss = sr_criterion(sr_out, hr_t)
                gsl_loss = gsl_criterion(gsl_out, source_pos)
                loss = args.sr_weight * sr_loss + args.gsl_weight * gsl_loss

                if args.enable_pred and hr_tp1 is not None:
                    pred_loss = pred_criterion(pred_out, hr_tp1)
                    loss = loss + args.pred_weight * pred_loss
                    pred_loss_sum += pred_loss.item()
                else:
                    pred_loss = torch.tensor(0.0, device=device)

                total_loss_sum += loss.item()
                sr_loss_sum += sr_loss.item()
                gsl_loss_sum += gsl_loss.item()

                postfix = {
                    "loss": f"{loss.item():.4f}",
                    "sr": f"{sr_loss.item():.4f}",
                    "gsl": f"{gsl_loss.item():.4f}",
                }
                if args.enable_pred and hr_tp1 is not None:
                    postfix["pred"] = f"{pred_loss.item():.4f}"
                else:
                    postfix["pred"] = "0.0"
                pbar.set_postfix(postfix)

        n_valid = len(valid_loader)
        valid_total = total_loss_sum / n_valid
        valid_sr = sr_loss_sum / n_valid
        valid_gsl = gsl_loss_sum / n_valid
        valid_pred = (pred_loss_sum /
                      n_valid) if (args.enable_pred and args.use_seq) else 0.0

        # scheduler
        scheduler.step(valid_total)

        # history append
        train_losses.append(train_total)
        valid_losses.append(valid_total)
        train_sr_losses.append(train_sr)
        valid_sr_losses.append(valid_sr)
        train_gsl_losses.append(train_gsl)
        valid_gsl_losses.append(valid_gsl)
        if args.enable_pred and args.use_seq:
            train_pred_losses.append(train_pred)
            valid_pred_losses.append(valid_pred)

        # epoch summary
        print(f"\nEpoch {epoch+1}/{args.num_epochs} Summary:")
        print(f"  Train total={train_total:.4f} | SR={train_sr:.4f} | GSL={train_gsl:.4f}"
              + (f" | Pred={train_pred:.4f}" if (args.enable_pred and args.use_seq) else ""))
        print(f"  Valid total={valid_total:.4f} | SR={valid_sr:.4f} | GSL={valid_gsl:.4f}"
              + (f" | Pred={valid_pred:.4f}" if (args.enable_pred and args.use_seq) else ""))
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # save best
        if valid_total < best_valid_loss:
            best_valid_loss = valid_total
            best_path = os.path.join(args.save_dir, "best_model_original.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "valid_loss": valid_total,
                "args": vars(args),
            }, best_path)
            print(f"Best model saved to {best_path}")

        # save latest checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "train_sr_losses": train_sr_losses,
            "valid_sr_losses": valid_sr_losses,
            "train_gsl_losses": train_gsl_losses,
            "valid_gsl_losses": valid_gsl_losses,
            "train_pred_losses": train_pred_losses,
            "valid_pred_losses": valid_pred_losses,
            "best_valid_loss": best_valid_loss,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.save_dir, "latest_checkpoint.pth"))

        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, os.path.join(args.save_dir,
                       f"checkpoint_epoch_{epoch+1}.pth"))
            print(f"Checkpoint saved at epoch {epoch+1}")

        # plots + csv
        plot_loss_lines(train_losses, valid_losses,
                        train_sr_losses, valid_sr_losses,
                        train_gsl_losses, valid_gsl_losses,
                        train_pred_losses, valid_pred_losses,
                        args.save_dir,
                        enable_pred=(args.enable_pred and args.use_seq))
        save_training_history(args.save_dir,
                              train_losses, valid_losses,
                              train_sr_losses, valid_sr_losses,
                              train_gsl_losses, valid_gsl_losses,
                              train_pred_losses, valid_pred_losses,
                              enable_pred=(args.enable_pred and args.use_seq))

    # save best copy
    best_path = os.path.join(args.save_dir, "best_model_original.pth")
    if os.path.exists(best_path):
        shutil.copy2(best_path, os.path.join(
            args.save_dir, "original_best_model.pth"))

    print("\nTraining completed!")
    print(f"Best validation loss: {best_valid_loss:.4f}")
    return model


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    # dataset
    if args.use_test_set:
        train_set, valid_set, test_set = generate_train_valid_test_dataset(
            args.data_path, train_ratio=0.8, valid_ratio=0.1, shuffle=True, seed=args.seed,
            K=args.K, use_seq_dataset=args.use_seq
        )
        print("Split: train/val/test = 0.8/0.1/0.1")
    else:
        train_set, valid_set, _ = generate_train_valid_test_dataset(
            args.data_path, train_ratio=0.8, valid_ratio=0.2, shuffle=True, seed=args.seed,
            K=args.K, use_seq_dataset=args.use_seq
        )
        print("Split: train/val = 0.8/0.2")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # sanity print
    batch = next(iter(train_loader))
    print("\n=== Batch keys ===", list(batch.keys()))
    if args.use_seq:
        print("lr_seq:", tuple(batch["lr_seq"].shape))
        print("hr_t:", tuple(batch["hr_t"].shape))
        print("hr_tp1:", tuple(batch["hr_tp1"].shape))
    else:
        print("lr:", tuple(batch["lr"].shape))
        print("hr:", tuple(batch["hr"].shape))
    print("source_pos:", tuple(batch["source_pos"].shape))

    # model
    model = create_model(args)

    # train
    train_model(model, train_loader, valid_loader, args)


if __name__ == "__main__":
    main()
