# プロジェクト進行ログ

## 2026-09-01: プロジェクト開始・環境構築

### 環境
- GPU: NVIDIA GeForce RTX 3090 (24GB) / Driver 595.71.05 / CUDA 13.2
- Python 3.10.12 (venv)
- 主要ライブラリ: torch 2.13.0+cu130 (CUDA有効), timm 1.0.29, gradio 6.26.0
  - その他: torchvision, scikit-learn, matplotlib, pandas, Pillow

### 作成したもの
- フォルダー構成: scripts / images / saved_models / docs / results
- `scripts/app.py` — Gradio UIアプリ本体
  - 学習タブ: 親フォルダー指定 → クラス自動検出（クラス名編集可）→ TIMMモデル選択 → RUN で転移学習
  - モデル: ConvNeXt-Tiny / ConvNeXt-Large / EfficientNet-B0 / EfficientNet-B4
  - 転移学習: バックボーン低学習率(×0.1)＋ヘッド高学習率のAdamW、AMP(混合精度)、ベストモデル自動保存
  - 推論タブ: 学習済みモデル選択 → 画像フォルダー一括推論 → 表＋ギャラリー表示、CSVをresults/へ保存
- `scripts/download_demo_data.sh` — デモデータ取得スクリプト
  - PlantVillageデータセット(GitHub)からリンゴの葉4クラスをsparse cloneで取得
  - クラス: 健康 / 黒星病(scab) / 黒腐病(black rot) / さび病(cedar apple rust)
  - 各クラス 学習80枚・テスト10枚 → `images/demo_leaves/{train,test}/`

### 動作検証（スモークテスト）
- ConvNeXt-Tiny・3エポックで学習 → **検証精度 100%**（Epoch1: 87.5% → Epoch3: 100%）
- さび病テスト画像10枚を推論 → **10枚全て正解**（確信度 96〜99.9%）
- UIアプリ起動確認済み (http://localhost:7860)

### 起動方法
```bash
cd /home/deeplion/claude_timm
venv/bin/python scripts/app.py
# → http://localhost:7860
```
