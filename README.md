# DiffSensei（WAIベース・ステージ2特化フォーク）

[![arXiv](https://img.shields.io/badge/arXiv-2412.07589-b31b1b.svg)](https://arxiv.org/abs/2412.07589)

本リポジトリは [DiffSensei](https://github.com/jianzongwu/DiffSensei)（arXiv:2412.07589, SDXL拡張の
「複数キャラを指定位置・レイアウトで一貫生成する白黒漫画パネル生成モデル」）のフォークです。

## このフォークの目的

**RTX 5060 Ti 16GB ×1 / Windows** 環境で、日本語同人誌（美少女・白黒漫画）向けに
DiffSensei を学習・推論することに絞り込んでいます。

- 作画は **WAI Illustrious SDXL** が既に高品質に描けるため、**ステージ1（t2i / 作画ドメイン適応）は学習しない**。
- 「指定キャラを・指定bbox領域に・一貫した見た目で描く」「吹き出し配置」を担う DiffSensei 独自モジュール
  （ステージ2：キャラ注入＋レイアウト制御）**のみを学習**する。
- **ステージ3（MLLM / SEED-X 17B）は 16GB に載らず非対応**。関連コード・参考config は削除/隔離済み。

採用ルート: WAIベースに公開 DiffSensei の IP/dialog モジュールを初期値として移植し、
`unet_trained_parameters: new` で WAI の作画空間へ再整合学習する。

## クイックスタート

前提: Python 3.12 / PyTorch 2.9 + CUDA 12.8 / Windows 11。

```powershell
pip install -r requirements.txt
```

学習・推論・データ準備の**具体的なコマンド**、アーキテクチャ詳細、Windows 固有の落とし穴、
16GB 最適化は **[CLAUDE.md](CLAUDE.md)** に集約しています。代表的な入口だけ示すと:

```powershell
# 学習データ準備（WAI変換 / 公開DiffSensei初期重み取得 / 自動アノテ / latent・text キャッシュ）
python -m scripts.train.prepare_wai
python -m scripts.train.prepare_diffsensei
python -m scripts.dataset.auto_annotate --image_root data --ann_path data/annotations/train.json --caption wd14

# ステージ2（condition）学習 — 単一GPU必須（accelerate launch は使わない）
python -m scripts.train.train --config_path configs/train/diffsensei/self_finetune_wai_condition_5060ti.yaml

# 推論 / 評価 / 参照立ち絵
python -m scripts.inference.inference_trained --config ... --ckpt ... --input_json configs/inference/eval_input.json
python -m scripts.eval.reproduce --config ... --ckpt ... --page 113
python -m scripts.refs.gen_wai --character "shigure ui (vtuber)" --n 5 --out refs/
```

共有パイプライン構築は `src/inference/pipeline.py`。

## 上流との差分

- 上流の Gradio デモ・MangaZero ダウンローダ・参考用ステージ1/3 学習コード・MLLM(SEED-X)推論コードは、
  本構成では不要なため削除しています。
- 上流の参考 config は `configs/_upstream/` に隔離しています。
- 上流の完全な利用方法は [本家リポジトリ](https://github.com/jianzongwu/DiffSensei) を参照してください。

## 引用

```
@article{wu2024diffsensei,
  title={DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation},
  author={Jianzong Wu, Chao Tang, Jingbo Wang, Yanhong Zeng, Xiangtai Li, and Yunhai Tong},
  journal={arXiv preprint arXiv:2412.07589},
  year={2024},
}
```
