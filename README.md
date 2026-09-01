# 🌿 TIMM 転移学習 画像分類ツール

[TIMM (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models) の事前学習済みモデルを使い、**少数の画像（各クラス30〜100枚程度）から転移学習で画像分類モデルを構築できる** Gradio UI アプリです。

植物の葉の病気判別をデモとして用意していますが、昆虫・植物など任意の分類タスクにそのまま使い回せます。

> このリポジトリは、植物学会における [Claude Code](https://claude.com/claude-code) のデモとして、Claude Code が自律的に構築したものです。

## ✨ 特徴

- **ブラウザUIだけで完結** — フォルダーを指定して RUN するだけ。コードを書く必要はありません
- **少数データでも高精度** — ImageNet事前学習済みモデルからの転移学習（バックボーン低学習率 + 分類ヘッド高学習率のAdamW、混合精度学習）
- **モデルを選べる** — ConvNeXt / EfficientNet の大小4種類
  | 表示名 | timmモデル | 用途 |
  |---|---|---|
  | ConvNeXt-Tiny | `convnext_tiny.fb_in22k_ft_in1k` | 小・高速（まずはこれ） |
  | ConvNeXt-Large | `convnext_large.fb_in22k_ft_in1k` | 大・高精度 |
  | EfficientNet-B0 | `efficientnet_b0.ra_in1k` | 小・高速 |
  | EfficientNet-B4 | `tf_efficientnet_b4.ns_jft_in1k` | 大・高精度 |
- **一括推論** — フォルダーを指定すると全画像を分類し、結果表・ギャラリー表示・CSV保存まで自動

## 📋 動作環境

- Python 3.10 以上
- NVIDIA GPU + CUDA を推奨（CPUでも動作しますが学習は低速です）
- 動作確認環境: RTX 3090 / CUDA 13.2 / torch 2.13.0

## 🚀 セットアップ

```bash
git clone <このリポジトリ>
cd claude_timm

# venv作成とライブラリインストール
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

## 📁 デモデータの準備

[PlantVillage データセット](https://github.com/spMohanty/PlantVillage-Dataset) からリンゴの葉4クラス（健康・黒星病・黒腐病・さび病）を、各クラス学習80枚＋テスト10枚だけダウンロードします。

```bash
bash scripts/download_demo_data.sh
# → images/demo_leaves/{train,test}/ に配置されます
```

## 🖥️ 使い方

```bash
venv/bin/python scripts/app.py
# → ブラウザで http://localhost:7860 を開く
```

### 学習タブ

1. **親フォルダー**を指定（例: `images/demo_leaves/train`）
   - 親フォルダー直下の「クラスごとのサブフォルダー」を自動検出します
2. 「📂 クラスを検出」を押す — クラス名は表の中で自由に編集できます
3. モデル・エポック数などを選んで「🚀 RUN」
4. 学習が終わると `saved_models/` にベストモデルが自動保存されます

```
images/demo_leaves/train/     ← ここを指定
├── healthy_健康/       ← サブフォルダー名 = クラス名（変更可）
├── scab_黒星病/
├── black_rot_黒腐病/
└── rust_さび病/
```

### 推論タブ

1. 「🔄 更新」で学習済みモデルを選択
2. 分類したい画像フォルダーを指定して「🔍 推論実行」
3. 結果は表＋ギャラリーで表示され、CSVが `results/` に自動保存されます

### 参考: デモでの性能

ConvNeXt-Tiny・各クラス80枚・3エポック（RTX 3090で約1分）で検証精度100%、テスト画像も全問正解でした。

## 📂 ファイル構成

```
.
├── scripts/
│   ├── app.py                    # Gradio UIアプリ本体（学習＋推論）
│   └── download_demo_data.sh     # デモデータ取得スクリプト
├── images/            # 画像データ置き場（デモデータもここに入る）
├── saved_models/      # 学習済みモデルの自動保存先（各モデル: model.pt + meta.json）
├── results/           # 推論結果CSVの出力先
├── docs/              # 進行ログなど
├── requirements.txt
└── CLAUDE.md          # Claude Code 向けプロジェクト指示書
```

※ `venv/`・画像データ・学習済みモデルはリポジトリに含まれません（`.gitignore` で除外）。

## 🔧 自分のデータで使うには

クラスごとのサブフォルダーを持つ親フォルダーを用意するだけです。

```
images/my_insects/
├── モンシロチョウ/   # 30〜100枚程度
├── アゲハチョウ/
└── オオムラサキ/
```

学習タブでこの親フォルダーを指定して RUN すれば、昆虫でも植物でも同じ手順で分類モデルが作れます。

## 📜 謝辞・ライセンス

- モデル: [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)
- デモデータ: [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
  - Hughes, D.P. & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060
- このリポジトリのコードは Claude Code (Anthropic) により作成されました
