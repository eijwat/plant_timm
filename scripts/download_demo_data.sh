#!/bin/bash
# デモ用データセット (PlantVillage リンゴの葉 4クラス) のダウンロード
# 学習用に各クラス80枚、テスト用に各クラス10枚を images/demo_leaves/ に配置する
set -e

PROJECT=/home/deeplion/claude_timm
TMP=$PROJECT/images/_pv_tmp
DEST=$PROJECT/images/demo_leaves

CLASSES=(
  "Apple___healthy"
  "Apple___Apple_scab"
  "Apple___Black_rot"
  "Apple___Cedar_apple_rust"
)
# 日本語のクラス名 (フォルダー名として使用)
NAMES=(
  "healthy_健康"
  "scab_黒星病"
  "black_rot_黒腐病"
  "rust_さび病"
)

echo "=== PlantVillage データセットを部分クローン中 ==="
rm -rf "$TMP"
git clone --filter=blob:none --sparse --depth 1 \
  https://github.com/spMohanty/PlantVillage-Dataset.git "$TMP"
cd "$TMP"
SPARSE_PATHS=()
for c in "${CLASSES[@]}"; do
  SPARSE_PATHS+=("raw/color/$c")
done
git sparse-checkout set "${SPARSE_PATHS[@]}"

echo "=== 学習80枚 / テスト10枚 に絞って配置 ==="
rm -rf "$DEST"
for i in "${!CLASSES[@]}"; do
  SRC="$TMP/raw/color/${CLASSES[$i]}"
  NAME="${NAMES[$i]}"
  mkdir -p "$DEST/train/$NAME" "$DEST/test/$NAME"
  # ファイル名でソートして先頭80枚を学習、次の10枚をテストに
  mapfile -t FILES < <(ls "$SRC" | sort | head -90)
  for j in "${!FILES[@]}"; do
    if [ "$j" -lt 80 ]; then
      cp "$SRC/${FILES[$j]}" "$DEST/train/$NAME/"
    else
      cp "$SRC/${FILES[$j]}" "$DEST/test/$NAME/"
    fi
  done
  echo "  $NAME: train=$(ls "$DEST/train/$NAME" | wc -l) test=$(ls "$DEST/test/$NAME" | wc -l)"
done

rm -rf "$TMP"
echo "=== 完了: $DEST ==="
