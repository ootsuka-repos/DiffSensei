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
- **`USE_LIBUV=0` は単一プロセス学習でも必須**: 未設定だと `Accelerator()` が分散用 TCPStore
  （libuv 依存）を初期化しようとして、**トレースバックも出さず起動直後に無言終了**する
  （import 警告の直後にプロンプトへ戻る＝これが症状）。学習起動前に必ず `$env:USE_LIBUV="0"`
  （bash は `USE_LIBUV=0`）を設定する。
- **マルチGPU不可**: Windows 版 PyTorch は libuv 非対応で `accelerate launch --multi_gpu` は
  `DistStoreError` で落ちる（こちらは `USE_LIBUV=0` でも直らない）。**必ず単一プロセスで実行**
  （`python -m scripts.train.train ...`、`accelerate launch` を使わない）。
- **DataLoader のワーカー起動が不安定**: Windows の spawn ベースのワーカーは、`Running training`
  到達直後（最初のバッチ取得時）に**トレースバックも出さず無言終了/ハング**することがある（日本語パスの
  データだと特に）。config で **`train_data.num_workers: 0`**（本体プロセスでロード）にするのが最も確実。
  少し遅くなるが安定する。
- **アノテーション JSON は UTF-8 で読む**: `data/<作品名>` が日本語だと JSON に日本語パスが入り、
  既定の cp932 デコードで `UnicodeDecodeError` になる。`dataset_size_bucket.py` は `encoding='utf-8'`
  で開く（修正済み）。新規に JSON を読むコードを足すときも必ず `encoding='utf-8'` を付ける。
- **パスのコロン禁止**: ログのタイムスタンプは `%Y-%m-%d-%H-%M-%S`（コロンは Windows で不正）。
- **`scripts` パッケージのシャドーイング**: site-packages 側に `scripts` がありローカルを隠すため、
  `scripts/`・`scripts/train/`・`scripts/dataset/`・`scripts/inference/`・`scripts/eval/`・`scripts/refs/` に
  **空の `__init__.py` が必要**（`python -m scripts.train.train` を成立させるため。消さないこと）。
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

# 4) VAE latent を事前計算してキャッシュ（use_latent_cache: true の前提。VAEを学習ループから外す）
#    ann / max_bucket_size / ベースVAE を変えたら必ず作り直す。
python -m scripts.train.precompute_latents --config configs/train/diffsensei/self_finetune_wai_condition_5060ti.yaml

# 5) テキスト埋め込みを事前計算してキャッシュ（use_text_cache: true の前提。テキストエンコーダ×2を外す ~1.6GB）
#    ann のキャプション / ベースのテキストエンコーダ を変えたら作り直す。
python -m scripts.train.precompute_text_embeds --config configs/train/diffsensei/self_finetune_wai_condition_5060ti.yaml
```

### ステージ2（condition）学習 — 現行の本命
**必ず単一GPU**（`accelerate launch` は使わない）。`USE_LIBUV=0` は `train.py` 先頭で自動設定する
ので、環境変数なしの素のコマンドだけで起動できる（事前に上記 4) の latent キャッシュを作成しておく）:
```powershell
python -m scripts.train.train --config_path configs/train/diffsensei/self_finetune_wai_condition_5060ti.yaml
```
必要に応じて任意で上書き可（GPU選択やオフライン強制）:
```powershell
$env:CUDA_VISIBLE_DEVICES="0"      # 使う GPU を固定したい場合
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"   # ネットワーク待ちを完全に断ちたい場合
```
- チェックポイント: `logs/diffsensei/self_finetune_wai_condition_5060ti_5060ti_wai_cond/<timestamp>/step-*/ckpt.pth`
  （`{"image_proj":..., "unet_trained":...}` の形で**学習した IP/dialog/Resampler のみ**保存）。
- ペース目安: 約5.5秒/step（暖機後）。2000 step ≈ 約3時間、300 step 毎に保存。
- 起動直後に `grafted DiffSensei IP/dialog: NNN tensors, unexpected=0` が出れば WAI へのグラフト成功。
- クイック確認は config の `max_train_steps` を一時的に小さくする。

### 推論
上流の Gradio デモは削除済み。本フォークでは役割別 CLI と `src/inference/` 共有ライブラリを使う。

**パイプライン構築（共有）**: `src/inference/pipeline.py` の `build_pipeline` / `resolve_weight_dtype` /
`load_ip_images`。学習チェックポイントから `DiffSenseiPipeline` を組み立てるロジックはここに集約。

```powershell
# 単一パネル推論（JSON 仕様は configs/inference/eval_input.json を参照）
python -m scripts.inference.inference_trained \
    --config configs/train/diffsensei/self_finetune_wai_condition_5060ti.yaml \
    --ckpt logs/.../epoch-1/ckpt.pth \
    --input_json configs/inference/eval_input.json --output_dir outputs --tag epoch1

# 学習データページの GT vs 生成比較（学習 eval も同モジュール）
python -m scripts.eval.reproduce \
    --config configs/train/diffsensei/self_finetune_wai_condition_5060ti.yaml \
    --ckpt logs/.../epoch-1/ckpt.pth --page 113 --out outputs/repro.png

# 複数コマを1ページに合成（セリフは PIL で後合成）
python -m scripts.inference.make_page \
    --config configs/train/diffsensei/self_finetune_wai_condition_5060ti.yaml \
    --ckpt logs/.../epoch-1/ckpt.pth \
    --spec configs/inference/eval_page.json --out outputs/page.png

# 参照キャラ立ち絵生成（WAI t2i、DiffSensei モジュール無し）
python -m scripts.refs.gen_wai --character "shigure ui (vtuber)" --n 5 --out refs/
```

要点:
- 構築: WAI ベース（`checkpoints/wai-illustrious-diffusers`）の `UNetMangaModel` に `set_manga_modules`
  → 学習 `ckpt.pth` の `unet_trained`(IP/dialog)＋`image_proj`(Resampler) をオーバーレイ。
- 入力: `prompt` / 解像度（8の倍数, 64〜2048）/ `ip_images` / `ip_bbox`（0〜1）/ `dialog_bbox`。
- eval サンプルは `scripts.eval.reproduce`（1280 タイア固定・bf16）。学習中は `train.py` が epoch 保存後に GPU1 で自動起動。
- 「キャラ単体イラスト」用途は `dialog_bbox` を渡さず、ネガティブに `speech bubble, text, comic` を加える。

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
| 学習済み推論（共有） | `build_pipeline` 等 / `src/inference/pipeline.py` |

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
- `optimizer.use_8bit_adam: true` ＋ `optimizer.paged: true`（`PagedAdamW8bit`：オプティマイザ状態を
  unified memory に置き VRAM スパイク時に CPU へページング＝OOM 耐性）。
- 部分学習（`new`）＋凍結重みの bf16 化。SDPA メモリ効率アテンション（`AttnProcessor2_0`）。
- `train_batch_size: 1` ＋ `gradient_accumulation_steps: 4`（実効バッチ4）。
- **VAE は bf16＋タイリング**（`train_data.vae_dtype: bf16`, `vae_tiling: true`）: 高解像度で最大の
  メモリ食いだった **fp32 VAE encode を bf16 化＋`enable_tiling()/enable_slicing()` で分割実行**して
  ピークを平坦化。bf16 は指数部が広く fp16-VAE の NaN を回避。
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**（`train.py` 先頭で自動設定）で断片化 OOM を抑制。
- **VAE latent 事前計算キャッシュ**（`train_data.use_latent_cache: true`／最大効果）: 学習対象はアダプター
  だけ＝ターゲットのコマ latent は (page,frame) ごとに**決定的**なので、`scripts.train.precompute_latents`
  で一度だけ encode して `data/latent_cache/wai_maxb<N>/<ann>_<frame>.pt`（mean/std/crop 座標）に保存。
  学習時は **VAE を GPU に載せず**、`latents = (mean + std·ε)·scaling_factor` をキャッシュからサンプルする
  （= fp32 VAE encode の活性化メモリと VAE 重みが丸ごと消える）。**ann・`max_bucket_size`・ベース VAE を
  変えたら作り直す**（キャッシュ dir 名は `max_bucket_size` を含む）。未生成だと `train.py` が起動時に
  明示エラーで `precompute_latents` を促す。
- **解像度 `train_data.max_bucket_size: 1280`**: SDXL はネイティブ 1024・高解像度 ~1280 が得意なので、
  小コマを 512 に潰さず**大コマは 1024〜1280 で学習**する。`size_buckets` は **タイアの正方形サイズ**
  （`size`: 256/512/1024/1280）で管理され、`train.py` は **`size <= max_bucket_size`** のタイアを残す
  （＝「最大辺」ではなく「正方形の一辺」でフィルタ）。各コマは自分の面積に最も近いタイアへ割り当て
  （小コマは小タイアのまま＝無駄な拡大をしない）。本データの分布: 256→24, 512→304, 1024→145, 1280→134 コマ。
  - 1280 タイア（`_make_area_tier(1280, ...)` で 1024 タイアと同じ 33 アスペクト比から生成）は
    `max_bucket_size >= 1280` のときだけ使われる。VAE/テキストはキャッシュで外に出ているので、1280 の
    VRAM 余地は **UNet forward 用**。
  - **OOM したら `max_bucket_size: 1024`（または 768）へ下げ、`precompute_latents` を再実行**して
    latent キャッシュを作り直す（キャッシュ dir 名は `max_bucket_size` を含む）。テキストキャッシュは
    解像度非依存なので作り直し不要。
  - ⚠ 旧仕様は「最大**辺** <= max_bucket_size」でフィルタしていたため、`1024` 指定でも実際は正方形 512 まで
    しか使えていなかった。現在は正方形サイズ基準なので `1024` = 1024×1024 で学習する。
- **テキスト埋め込みキャッシュ**（`train_data.use_text_cache: true`／約1.6GB 解放）: latent と同様、caption は
  決定的で `t_drop` で空文字に落ちるだけなので、`scripts.train.precompute_text_embeds` で各コマの
  `text_embeds[77,2048]＋pooled[1280]` と空文字版を `data/text_cache/wai/` に保存。学習時は**テキスト
  エンコーダ×2 を GPU に載せず**、`t_drop` に応じてキャッシュを引く。captions/ベースのテキストエンコーダを
  変えたら作り直す。
- 未実装の追加策（必要なら）: **画像エンコーダ（CLIP/Magi）出力のキャッシュ**は self-condition・ランダム
  flip・全フレーム横断のソース選択でキャッシュキーが複雑になりリスクが高いため見送り。凍結 UNet の int8
  量子化（約2.3GB 解放だがアダプター混在で実装リスク高）も同様に未実装。

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
- 事前計算キャッシュ: `scripts/train/precompute_latents.py`（VAE latent）, `precompute_text_embeds.py`（テキスト埋め込み）
- 参照キャラ画像（IP refs・永続）: `refs/work1〜5.png`（WAI生成のしぐれうい立ち絵。`outputs/` ではなく `refs/` に置く）
- アノテーション: `scripts/dataset/auto_annotate.py`
- 推論エンジン: `src/pipelines/pipeline_diffsensei.py`（`DiffSenseiPipeline`）
- 学習済み推論（共有）: `src/inference/`（`pipeline.py`, `eval_utils.py`）
- 推論 CLI: `scripts/inference/inference_trained.py`, `make_page.py`
- 評価 CLI: `scripts/eval/reproduce.py`（学習 epoch サンプル生成）
- 参照画像生成: `scripts/refs/gen_wai.py`
- 推論フィクスチャ: `configs/inference/eval_input.json`, `eval_page.json`
- 現行学習 config: `configs/train/diffsensei/self_finetune_wai_condition_5060ti.yaml`
- 参考 config（元リポジトリ）: `configs/train/diffsensei/self_0.5.yaml`,
  `configs/model/diffsensei.yaml`, `configs/inference/diffsensei.yaml`
