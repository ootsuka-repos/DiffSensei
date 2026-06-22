# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 言語: このリポジトリでのやり取り・コメント・説明はすべて**日本語**で行う。

---

## このリポジトリは何か

**DiffSensei**（arXiv:2412.07589）= SDXL を拡張した「複数キャラを指定位置・レイアウトで一貫生成する
白黒漫画パネル生成モデル」のフォーク。本フォークの主目的は、**ローカルの RTX 5060 Ti 16GB ×1 /
Windows 環境で、日本語同人誌（美少女・白黒漫画）向けに学習・推論を動かす**こと。

### 現段階の確定方針（最重要）
**必要な学習はステージ2（キャラ注入＋レイアウト制御）だけ**。理由:
- 作画は **WAI Illustrious SDXL**（`checkpoints/waiIllustriousSDXL_v170.safetensors`）が既に高品質に
  描けるので、**ステージ1（t2i / 作画ドメイン適応）は学習しない**。
- 「指定キャラを・指定bbox領域に・一貫した見た目で描く」「吹き出し配置」は DiffSensei 独自モジュール
  が担い、SDXL/WAI には無いので**ステージ2は学習が必要（核）**。
- ステージ3（MLLM / SEED-X 17B）は 16GB に載らず**非対応**。

採用ルート: **WAIベース＋公開DiffSenseiのIP/dialogモジュールを初期値に移植し、`unet_trained_parameters:
new` で再整合学習**。過去検証で「移植しただけ→IP出力が破綻」「DiffSenseiベース→作画が英語漫画寄り」と
判明したため、両者の良いとこ取り（WAIの作画＋整合したキャラ注入）を狙う。

---

## 環境と Windows 固有の落とし穴

- Windows 11 / Python 3.12 / PyTorch 2.9 + CUDA 12.8 / RTX 5060 Ti 16GB ×1。
- **マルチGPU不可**: Windows 版 PyTorch は libuv 非対応で `accelerate launch --multi_gpu` が
  `DistStoreError` で落ちる（`USE_LIBUV=0` でも直らない）。**必ず単一プロセスで実行**
  （`python -m scripts.train.train ...`、`accelerate launch` を使わない）。
- **DataLoader ハング**: `num_workers` が多い（既定8）と Windows でハングする。config で
  `train_data.num_workers: 2` を指定（`train.py` は `persistent_workers` 付きで尊重する）。
- **パスのコロン禁止**: ログのタイムスタンプは `%Y-%m-%d-%H-%M-%S`（コロンは Windows で不正）。
- **`scripts` パッケージのシャドーイング**: site-packages 側に `scripts` がありローカルを隠すため、
  `scripts/`・`scripts/train/`・`scripts/dataset/`・`scripts/demo/` に**空の `__init__.py` が必要**
  （`python -m scripts.train.train` を成立させるため。消さないこと）。
- **torchao**: PEFT の LoRA dispatch には torchao > 0.16 が必要（`pip install -U torchao --no-deps`）。

---

## よく使うコマンド

すべて**単一GPU**・リポジトリルートから実行。学習設定が HF キャッシュ依存のため、オフライン実行時は
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` を付ける（DL が要る準備ステップでは付けない）。

### 学習データ準備
```bash
# 1) WAI 単一ファイル → diffusers レイアウトへ変換（作画ベース）
python -m scripts.train.prepare_wai          # -> checkpoints/wai-illustrious-diffusers

# 2) 公開 DiffSensei の IP/dialog 初期重みを取得（unet 11.6GB + image_proj 336MB、curl 再開対応）
python -m scripts.train.prepare_diffsensei   # -> checkpoints/diffsensei/image_generator

# 3) data/ の生ページ画像を Magi+WD14 で自動アノテーション
python -m scripts.dataset.auto_annotate --image_root data --ann_path data/annotations/train.json --caption wd14
```

### ステージ2（condition）学習 — 現行の本命
```bash
python -m scripts.train.train --config_path configs/train/diffsensei/self_finetune_wai_condition_5060ti.yaml
```
- チェックポイント: `logs/diffsensei/<config>_<exp_name>/<timestamp>/step-*/ckpt.pth`
  （`{"image_proj":..., "unet_trained":...}` の形で**学習した IP/dialog/Resampler のみ**保存）。
- クイック確認は config の `max_train_steps` を一時的に小さくする。

### 推論
デモ用 CLI/Gradio スクリプト（旧 `scripts/demo/`）は削除済み。推論は **`DiffSenseiPipeline`
（`src/pipelines/pipeline_diffsensei.py`）を直接呼ぶ薄いスクリプトを書いて実行**する。要点:
- 構築: WAI ベース（`checkpoints/wai-illustrious-diffusers`）の `UNetMangaModel` に `set_manga_modules`
  → 学習 `ckpt.pth` の `unet_trained`(IP/dialog)＋`image_proj`(Resampler) をオーバーレイ。
  画像エンコーダは `h94/IP-Adapter` の CLIP-ViT-H、キャラ大域特徴は Magi（`crop_embedding_model`）。
- 入力: `prompt` / 解像度（8の倍数, 64〜2048）/ `ip_images`（参照キャラ）/ `ip_bbox`（0〜1正規化）/
  `dialog_bbox`。`__call__` は `ip_scale` でキャラ注入強度を調整。
- 「キャラ単体イラスト（吹き出し無し）」用途は、`dialog_bbox` を渡さず、ネガティブに
  `speech bubble, text, comic` を加える。作画は凍結 WAI なので綺麗なイラストが出る。

---

## アーキテクチャ（big picture）

```
参照キャラ画像 ─[CLIP Vision]─┐
                              ├─[Resampler]→ image_embeds ─┐
参照キャラ画像 ─[Magi ViTMAE]─┘                            ├→ prompt 埋め込み末尾に連結
プロンプト ─[CLIP TextEnc×2]→ prompt_embeds ───────────────┘      │
IP bbox ────────────────────→ MaskedIPAttnProcessor（領域マスク注入）│
dialog bbox ────────────────→ UNet conv_in 直後に空間埋め込み加算    ↓
                                          [UNetMangaModel(SDXL拡張)] → ノイズ予測 → VAE → 画像
```

| 役割 | クラス / ファイル |
| --- | --- |
| 拡散パイプライン全体 | `DiffSenseiPipeline` / `src/pipelines/pipeline_diffsensei.py` |
| UNet（SDXL拡張・dialog埋め込み） | `UNetMangaModel` / `src/models/unet.py`（`set_manga_modules`, `encode_dialog_bbox`） |
| キャラ注入アテンション（核） | `MaskedIPAttnProcessor2_0` / `src/models/attention_processor.py` |
| 画像特徴の圧縮器 | `Resampler`（Perceiver型）/ `src/models/resampler.py` |
| MLLM（学習はしないが推論パイプラインが import） | `ContinuousLVLM` / `src/models/mllm/seed_x.py` |
| データセット | `src/datasets/dataset_size_bucket.py`（アスペクト比バケット可変解像度） |

### 押さえるべき設計原理
- **デカップルド・クロスアテンション**: `prompt_embeds = cat(text, image_embeds)` を渡し、プロセッサ内で
  末尾の image 部分を切り離し、専用 `to_k_ip`/`to_v_ip`（初期値はテキスト側 K/V クローン）で別計算し
  `hidden + ip_scale * ip_hidden` で加算。テキスト条件を壊さずキャラ特徴を後付け注入する。
- **空間マスク注入**: bbox を正規化座標で保持し、各 UNet 解像度ごとにマスクを再計算。「キャラAのトークンは
  キャラAの bbox 内ピクセルにしか効かない」を実現（複数キャラの干渉防止＋可変解像度 64〜2048 対応）。
- **ダミートークン**（`num_dummy_tokens=16`）: どの bbox にも属さない背景ピクセルの注目先。特徴漏れを抑制。
- **二系統エンコーダ**: CLIP=パッチ単位の局所ディテール、Magi=漫画キャラとしての大域的同一性。
- **dialog bbox 埋め込み**: 1本の学習ベクトルを `conv_in` 直後に bbox 領域へ加算。「ここは吹き出し用に余白を
  作る」という空間ヒントを最初期に与えるだけのシンプルな機構。
- **形状例**: `image_embeds = [1, num_dummy(16) + max_num_ips(4)*num_vision_tokens(16), cross_attn_dim(2048)] = [1,80,2048]`。

### 学習対象の切り替え（`model.unet_trained_parameters`）
`train.py` が対応: `full`（UNet全体）/ `lora`（LoRA を `_ip` 以外凍結の上で付与）/ `new`（`*_ip`＋`dialog`）/
`ip`（`*_ip` のみ）。本構成は **`new`**。`new`/`ip` では非学習パラメータを `requires_grad=False` にし
bf16 へキャストして VRAM 節約（学習対象は fp32 維持）。`image_proj`(Resampler) は常に学習される。

### WAIグラフト初期化（`train.py` 独自拡張）
`model.diffsensei_pretrained_path` 指定時、`set_manga_modules` 後に公開 DiffSensei の `unet/pytorch_model.bin`
を読み込み、`diffsensei_ip_only: true` なら `"_ip"`/`"dialog"` を含むキーだけ `load_state_dict(strict=False)`
で移植（作画は WAI のまま）。`image_proj_model/pytorch_model.bin` があれば Resampler も初期化する。

### ステージ2で「具体的に何を学習し、何が良くなるのか」
`new` 構成で学習する重みは次の3つだけ（**作画を担う WAI 本体・VAE・テキストエンコーダは凍結**）。
それぞれが別々の役割を持ち、「学習しないと何が壊れるか」で効果を理解できる。

| 学習する重み | 場所 | 何を学ぶか | 効果（＝学習しないとどうなるか） |
|---|---|---|---|
| **Resampler** `image_proj_model` | `src/models/resampler.py` | 参照キャラ画像の CLIP＋Magi 特徴を、UNet が読める**固定16トークンのキャラ表現**へ圧縮する変換 | 参照画像から「そのキャラらしさ」を正しく抽出できる。学習が弱いと**別人・崩れた顔**になる |
| **IPクロスアテンションの `to_k_ip` / `to_v_ip`** | `MaskedIPAttnProcessor2_0`（全クロスアテンション層） | キャラトークンを **WAI の作画特徴へ「描き込む」**ための鍵/値射影。初期値は移植した DiffSensei 重み→**WAI の作画空間へ再整合** | 「指定キャラを・指定 bbox 領域に・一貫した見た目で」描き込める。**未学習だと WAI と不整合で IP 出力が破綻**（＝過去にWAI移植だけで起きた現象）。これを直すのが本学習の主目的 |
| **dialog bbox 埋め込み** `dialog_bbox_embedding` | `UNetMangaModel.encode_dialog_bbox` | `dialog_bbox` で指定した矩形に「ここは吹き出し用の**余白**」という空間ヒントを刻む1本のベクトル | 吹き出し位置にキャラの絵がかぶらず、**セリフ用の空白枠**が自然に空く。※文字自体は描かない（後処理で合成） |

要するにステージ2は **「キャラの見た目を抽出する（Resampler）」「それを指定領域に描き込む（IPアテンション）」
「吹き出し用の余白を空ける（dialog 埋め込み）」** の3点を、**WAI の作画はそのままに**学習する。
作画品質は WAI 由来で既に高いので、ここでは触らず「キャラ制御・レイアウト制御」だけを上乗せするイメージ。

> 補足: 効果は推論時に `ip_scale`（キャラ注入の強さ）で調整できる。`ip_bbox` が配置領域、
> `dialog_bbox` が吹き出し領域。キャラ単体イラストが欲しければ `dialog_bbox` を渡さなければよい。

---

## 16GB に収めるための最適化（config で有効化済み）
- `mixed_precision: bf16` ＋ `gradient_checkpointing: true`。
- `optimizer.use_8bit_adam: true`（bitsandbytes `AdamW8bit`）。
- 部分学習（`new`）＋凍結重みの bf16 化。
- `train_batch_size: 1` ＋ `gradient_accumulation_steps: 4`（実効バッチ4）。

---

## 自動アノテーションの仕様（`scripts/dataset/auto_annotate.py`）
各ページに Magi（`ragavsachdeva/magi`）を実行し DiffSensei 形式 JSON を生成:
- パネル → `frames`（無ければページ全体を1コマ）。
- キャラ検出＋クラスタリングで一貫 `id` → `characters`（`type:0`）。同一人物は別パネルでも同 id。
- テキスト枠 → `dialogs`（中心点でパネルに割当）。
- パネル切り抜きに WD14 タガー → `manga, monochrome, greyscale, <tags>` のキャプション。
- 座標はページピクセル基準で保存（学習時にデータセットがパネル相対 0〜1 へ変換）。

出力1ページの例:
```json
{ "image_path": "1/01.webp",
  "frames": [{ "bbox": [1,2,560,420], "caption": "manga, monochrome, greyscale, 1girl, ...",
    "characters": [{"id":0, "bbox":[449,206,560,416], "type":0}], "dialogs": [{"bbox":[413,177,537,208]}] }] }
```

---

## 使用モデル（HF キャッシュ or 取得）
| 役割 | モデル | 取得元 |
| --- | --- | --- |
| 作画ベース | WAI Illustrious SDXL → diffusers 変換 | `checkpoints/wai-illustrious-diffusers` |
| IP/dialog 初期重み | `jianzongwu/DiffSensei`（`image_generator` のみ） | curl 取得（MLLM/SEED-X は不要なのでDLしない） |
| IP 画像エンコーダ | `h94/IP-Adapter`（CLIP-ViT-H） | HF キャッシュ |
| キャラ特徴＋アノテーション | `ragavsachdeva/magi` | HF キャッシュ |
| キャプション | WD14（`imgutils`） | ローカル |

`scripts/train/hf_paths.py` が HF キャッシュからこれらの実パスを解決する。

---

## 主要ファイル早見
- 学習（ステージ2のみ）: `scripts/train/train.py`。※ステージ1(t2i)/3(mllm)の学習スクリプト・configは
  本構成では未使用のため削除済み。
- 準備: `scripts/train/prepare_wai.py`, `prepare_diffsensei.py`, `hf_paths.py`
- アノテーション: `scripts/dataset/auto_annotate.py`
- 推論エンジン: `src/pipelines/pipeline_diffsensei.py`（`DiffSenseiPipeline`）。デモ CLI/Gradio は削除済み
  （必要時に薄い呼び出しスクリプトを書く）。
- 現行学習 config: `configs/train/diffsensei/self_finetune_wai_condition_5060ti.yaml`
- 参考 config（元リポジトリ）: `configs/train/diffsensei/self_0.5.yaml`,
  `configs/model/diffsensei.yaml`, `configs/inference/diffsensei.yaml`
