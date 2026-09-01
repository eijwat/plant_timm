# -*- coding: utf-8 -*-
"""
TIMM 転移学習による画像分類ツール (Gradio UI)

使い方:
  1. [学習] タブでクラスごとの画像フォルダーを含む親フォルダーを指定
  2. TIMMモデル (ConvNeXt / EfficientNet の大小) を選択して RUN
  3. [推論] タブで学習済みモデルと画像フォルダーを指定して分類

起動:
  venv/bin/python scripts/app.py
"""

import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import timm
from timm.data import resolve_data_config, create_transform

# ---------------------------------------------------------------
# 定数
# ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"
SAVED_MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用できるTIMMモデル (表示名: timmモデル名)
MODEL_CHOICES = {
    "ConvNeXt-Tiny (小・高速)": "convnext_tiny.fb_in22k_ft_in1k",
    "ConvNeXt-Large (大・高精度)": "convnext_large.fb_in22k_ft_in1k",
    "EfficientNet-B0 (小・高速)": "efficientnet_b0.ra_in1k",
    "EfficientNet-B4 (大・高精度)": "tf_efficientnet_b4.ns_jft_in1k",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def list_images(folder: Path):
    """フォルダー内の画像ファイルを列挙する"""
    return sorted(
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


# ---------------------------------------------------------------
# データセット
# ---------------------------------------------------------------
class ImageListDataset(Dataset):
    """(画像パス, ラベル) のリストから作るデータセット"""

    def __init__(self, samples, transform):
        self.samples = samples  # [(path, label_idx), ...]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ---------------------------------------------------------------
# 学習タブのロジック
# ---------------------------------------------------------------
def scan_class_folders(parent_dir: str):
    """親フォルダー直下のサブフォルダーをクラスとして検出し、表を返す"""
    if not parent_dir or not Path(parent_dir).is_dir():
        return (
            gr.update(value=pd.DataFrame(columns=["フォルダー", "クラス名", "画像枚数"])),
            "❌ フォルダーが見つかりません。パスを確認してください。",
        )
    parent = Path(parent_dir)
    rows = []
    for sub in sorted(p for p in parent.iterdir() if p.is_dir()):
        n = len(list_images(sub))
        if n > 0:
            rows.append([sub.name, sub.name, n])
    if not rows:
        return (
            gr.update(value=pd.DataFrame(columns=["フォルダー", "クラス名", "画像枚数"])),
            "❌ 画像を含むサブフォルダーが見つかりませんでした。",
        )
    df = pd.DataFrame(rows, columns=["フォルダー", "クラス名", "画像枚数"])
    total = sum(r[2] for r in rows)
    return (
        gr.update(value=df),
        f"✅ {len(rows)} クラス・合計 {total} 枚を検出しました。"
        "クラス名は表の中で編集できます。",
    )


def train_model(
    parent_dir: str,
    class_table: pd.DataFrame,
    model_display_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_ratio: float,
    progress=gr.Progress(),
):
    """転移学習を実行して saved_models/ に保存する"""
    # --- 入力チェック ---
    if not parent_dir or not Path(parent_dir).is_dir():
        return "❌ 学習データの親フォルダーを指定してください。", None
    if class_table is None or len(class_table) < 2:
        return "❌ クラスが2つ以上必要です。先に「クラスを検出」を押してください。", None

    parent = Path(parent_dir)
    model_name = MODEL_CHOICES[model_display_name]

    # --- クラス表からサンプルを収集 ---
    class_names = []
    samples = []
    for _, row in class_table.iterrows():
        folder = parent / str(row["フォルダー"])
        cname = str(row["クラス名"]).strip() or str(row["フォルダー"])
        if not folder.is_dir():
            continue
        imgs = list_images(folder)
        if not imgs:
            continue
        label = len(class_names)
        class_names.append(cname)
        samples.extend((p, label) for p in imgs)

    if len(class_names) < 2:
        return "❌ 有効なクラスが2つ未満です。", None

    # --- train/val 分割 (クラスごとに層化) ---
    rng = random.Random(42)
    train_samples, val_samples = [], []
    for lbl in range(len(class_names)):
        cls_samples = [s for s in samples if s[1] == lbl]
        rng.shuffle(cls_samples)
        n_val = max(1, int(len(cls_samples) * val_ratio))
        val_samples.extend(cls_samples[:n_val])
        train_samples.extend(cls_samples[n_val:])

    # --- モデルと変換 ---
    progress(0, desc="モデルをダウンロード/構築中...")
    model = timm.create_model(
        model_name, pretrained=True, num_classes=len(class_names)
    )
    config = resolve_data_config({}, model=model)
    train_tf = create_transform(**config, is_training=True)
    val_tf = create_transform(**config, is_training=False)
    model = model.to(DEVICE)

    train_ds = ImageListDataset(train_samples, train_tf)
    val_ds = ImageListDataset(val_samples, val_tf)
    train_dl = DataLoader(
        train_ds, batch_size=int(batch_size), shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=int(batch_size), shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # --- 最適化: バックボーンは低学習率、分類ヘッドは高学習率 ---
    head_params, body_params = [], []
    classifier = model.get_classifier()
    classifier_param_ids = {id(p) for p in classifier.parameters()}
    for p in model.parameters():
        (head_params if id(p) in classifier_param_ids else body_params).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": body_params, "lr": learning_rate * 0.1},
            {"params": head_params, "lr": learning_rate},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs)
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    # --- 学習ループ ---
    best_acc = 0.0
    log_lines = [
        f"モデル: {model_display_name} ({model_name})",
        f"クラス: {', '.join(class_names)}",
        f"学習 {len(train_samples)} 枚 / 検証 {len(val_samples)} 枚",
        f"デバイス: {DEVICE}",
        "-" * 50,
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = SAVED_MODELS_DIR / f"{model_name.split('.')[0]}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(epochs)):
        # 学習
        model.train()
        total_loss, n_seen = 0.0, 0
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
            progress(
                (epoch + (i + 1) / len(train_dl)) / int(epochs),
                desc=f"Epoch {epoch + 1}/{int(epochs)} 学習中...",
            )
        scheduler.step()
        train_loss = total_loss / max(n_seen, 1)

        # 検証
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(total, 1)
        log_lines.append(
            f"Epoch {epoch + 1:2d}/{int(epochs)}  "
            f"loss={train_loss:.4f}  val_acc={val_acc:.4f}"
        )

        # ベストモデルを保存
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_dir / "model.pt")

    # --- メタデータ保存 ---
    meta = {
        "model_name": model_name,
        "model_display_name": model_display_name,
        "class_names": class_names,
        "best_val_acc": best_acc,
        "epochs": int(epochs),
        "trained_at": timestamp,
        "data_dir": str(parent),
    }
    (save_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log_lines.append("-" * 50)
    log_lines.append(f"✅ 学習完了！ ベスト検証精度: {best_acc:.4f}")
    log_lines.append(f"保存先: {save_dir}")
    return "\n".join(log_lines), gr.update(choices=list_saved_models(), value=save_dir.name)


# ---------------------------------------------------------------
# 推論タブのロジック
# ---------------------------------------------------------------
def list_saved_models():
    """saved_models/ 内の学習済みモデル一覧"""
    return sorted(
        (d.name for d in SAVED_MODELS_DIR.iterdir()
         if d.is_dir() and (d / "meta.json").exists()),
        reverse=True,
    )


def refresh_models():
    choices = list_saved_models()
    return gr.update(choices=choices, value=choices[0] if choices else None)


def predict_folder(model_dir_name: str, image_dir: str, progress=gr.Progress()):
    """フォルダー内の画像を一括推論する"""
    if not model_dir_name:
        return None, None, "❌ 学習済みモデルを選択してください。"
    if not image_dir or not Path(image_dir).is_dir():
        return None, None, "❌ 画像フォルダーが見つかりません。"

    save_dir = SAVED_MODELS_DIR / model_dir_name
    meta = json.loads((save_dir / "meta.json").read_text(encoding="utf-8"))
    class_names = meta["class_names"]

    imgs = list_images(Path(image_dir))
    if not imgs:
        return None, None, "❌ フォルダーに画像がありません。"

    progress(0, desc="モデルを読み込み中...")
    model = timm.create_model(
        meta["model_name"], pretrained=False, num_classes=len(class_names)
    )
    model.load_state_dict(
        torch.load(save_dir / "model.pt", map_location=DEVICE, weights_only=True)
    )
    model = model.to(DEVICE).eval()
    config = resolve_data_config({}, model=model)
    tf = create_transform(**config, is_training=False)

    rows, gallery = [], []
    with torch.no_grad():
        for i, p in enumerate(imgs):
            img = Image.open(p).convert("RGB")
            x = tf(img).unsqueeze(0).to(DEVICE)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                probs = torch.softmax(model(x).float(), dim=1)[0]
            conf, idx = probs.max(0)
            rows.append(
                [p.name, class_names[idx.item()], f"{conf.item() * 100:.1f}%"]
            )
            gallery.append(
                (str(p), f"{class_names[idx.item()]} ({conf.item() * 100:.1f}%)")
            )
            progress((i + 1) / len(imgs), desc=f"推論中... {i + 1}/{len(imgs)}")

    df = pd.DataFrame(rows, columns=["ファイル名", "推定クラス", "確信度"])

    # CSVをresults/に保存
    csv_path = RESULTS_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # クラスごとの集計
    counts = df["推定クラス"].value_counts()
    summary = " / ".join(f"{c}: {n}枚" for c, n in counts.items())
    msg = f"✅ {len(imgs)} 枚を推論しました。（{summary}）\n結果CSV: {csv_path}"
    return df, gallery, msg


# ---------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------
with gr.Blocks(title="TIMM画像分類ツール") as demo:
    gr.Markdown("# 🌿 TIMM 転移学習 画像分類ツール")

    # ========== 学習タブ ==========
    with gr.Tab("学習"):
        gr.Markdown(
            "クラスごとのサブフォルダーを含む**親フォルダー**を指定してください。\n"
            "例: `images/demo_leaves/train` （その下に `healthy/`, `rust/` など）"
        )
        with gr.Row():
            train_dir_box = gr.Textbox(
                label="学習データの親フォルダー",
                placeholder=str(PROJECT_ROOT / "images" / "demo_leaves" / "train"),
                scale=4,
            )
            scan_btn = gr.Button("📂 クラスを検出", scale=1)
        class_table = gr.Dataframe(
            headers=["フォルダー", "クラス名", "画像枚数"],
            datatype=["str", "str", "number"],
            interactive=True,
            label="クラス一覧（クラス名は編集可）",
        )
        scan_msg = gr.Markdown("")

        with gr.Row():
            model_dd = gr.Dropdown(
                choices=list(MODEL_CHOICES.keys()),
                value="ConvNeXt-Tiny (小・高速)",
                label="TIMMモデル",
            )
            epochs_sl = gr.Slider(1, 50, value=10, step=1, label="エポック数")
            batch_sl = gr.Slider(4, 64, value=16, step=4, label="バッチサイズ")
        with gr.Row():
            lr_num = gr.Number(value=1e-3, label="学習率（ヘッド側）")
            val_sl = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="検証データ割合")

        run_btn = gr.Button("🚀 RUN（学習開始）", variant="primary")
        train_log = gr.Textbox(label="学習ログ", lines=16)

    # ========== 推論タブ ==========
    with gr.Tab("推論"):
        with gr.Row():
            model_select = gr.Dropdown(
                choices=list_saved_models(),
                label="学習済みモデル",
                scale=4,
            )
            refresh_btn = gr.Button("🔄 更新", scale=1)
        infer_dir_box = gr.Textbox(
            label="推論する画像フォルダー",
            placeholder=str(PROJECT_ROOT / "images" / "demo_leaves" / "test"),
        )
        infer_btn = gr.Button("🔍 推論実行", variant="primary")
        infer_msg = gr.Markdown("")
        with gr.Row():
            result_df = gr.Dataframe(label="推論結果", scale=1)
            result_gallery = gr.Gallery(label="画像と推定クラス", columns=4, scale=2)

    # ========== イベント ==========
    scan_btn.click(scan_class_folders, [train_dir_box], [class_table, scan_msg])
    run_btn.click(
        train_model,
        [train_dir_box, class_table, model_dd, epochs_sl, batch_sl, lr_num, val_sl],
        [train_log, model_select],
    )
    refresh_btn.click(refresh_models, None, [model_select])
    infer_btn.click(
        predict_folder, [model_select, infer_dir_box],
        [result_df, result_gallery, infer_msg],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
