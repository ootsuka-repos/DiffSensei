# DiffSensei: マルチモーダルLLMと拡散モデルを橋渡しするカスタマイズ漫画生成

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2410.08261-b31b1b.svg)](https://arxiv.org/abs/2412.07589)
[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=github-pages)](https://jianzongwu.github.io/projects/diffsensei)
[![Video](https://img.shields.io/badge/YouTube-Video-FF0000?logo=youtube)](https://www.youtube.com/watch?v=TLJ0MYZmoXc&source_ve_path=OTY3MTQ)
[![Checkpoint](https://img.shields.io/badge/🤗%20Huggingface-Model-yellow)](https://huggingface.co/jianzongwu/DiffSensei)
[![Dataset](https://img.shields.io/badge/🤗%20Huggingface-Dataset-yellow)](https://huggingface.co/datasets/jianzongwu/MangaZero)


</div>

デモは[プロジェクトページ](https://jianzongwu.github.io/projects/diffsensei)で公開しています。

## 🚀 TL;DR

DiffSensei は、柔軟なキャラクター適応を備えた、制御可能な白黒漫画パネルを生成できます。

**主な特徴:**
- 🌟 可変解像度の漫画パネル生成（辺の長さ 64〜2048 まで対応！）
- 🖼️ 1枚の入力キャラクター画像から、さまざまな見た目を生成
- ✨ 多彩な応用: カスタマイズ漫画生成、実写人物からの漫画制作


## 🎉 ニュース

- [2025-2-5] 参考用の学習コードを公開しました（t2i + condition + mllm）！
- [2024-12-13] MLLM を使わない新バージョンの Gradio デモを公開しました（メモリ使用量を大幅削減）！
- [2024-12-10] チェックポイント、データセット、推論コードを公開しました！

## 🛠️ クイックスタート

### インストール

``` bash
# Conda で新しい環境を作成
conda create -n diffsensei python=3.11
conda activate diffsensei
# PyTorch と Diffusers 関連パッケージをインストール
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge diffusers transformers accelerate
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
# その他の依存関係をインストール
pip install -r requirements.txt
# Gradio デモ実行用のサードパーティ製リポジトリ
pip install gradio-image-prompter
```

### モデルのダウンロード

DiffSensei モデルを [huggingface](https://huggingface.co/jianzongwu/DiffSensei) からダウンロードし、以下のように `checkpoints` フォルダに配置してください。

MLLM コンポーネントを使わない場合は、MLLM なしのモデルをダウンロードし、`gradio_wo_mllm.py`（または後述の `inference.py`）で結果を生成できます。

```
checkpoints
  |- diffsensei
    |- image_generator
      |- ...
    |- mllm
      |- ...
```


### このフォークでの学習・推論・運用

このフォークは **RTX 5060 Ti 16GB ×1 / Windows** で、日本語同人誌（美少女・白黒漫画）向けに
**ステージ2（キャラ注入＋レイアウト制御）のみ**を学習する構成に絞り込んでいます
（作画は WAI Illustrious ベースに任せ、ステージ1/3 の学習スクリプトは削除済み）。

- データ準備（WAI変換 / 公開DiffSensei IP重み取得 / Magi自動アノテーション）
- ステージ2 学習（`scripts/train/train.py` ＋ `self_finetune_wai_condition_5060ti.yaml`）
- 推論（`DiffSenseiPipeline` を直接呼ぶ）

の具体的なコマンド・アーキテクチャ詳細・Windows 固有の注意点（マルチGPU不可・DataLoaderハング等）・
16GB 最適化は、すべて **[CLAUDE.md](CLAUDE.md)** に集約しています。

> 上流オリジナルの推論（Gradio/CLI デモ）・MangaZero ダウンローダ・参考用ステージ1/3 学習コードは、
> 本構成では不要なため削除しています。上流の完全な利用方法は
> [本家リポジトリ](https://github.com/jianzongwu/DiffSensei) を参照してください。


## 引用

```
@article{wu2024diffsensei,
  title={DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation},
  author={Jianzong Wu, Chao Tang, Jingbo Wang, Yanhong Zeng, Xiangtai Li, and Yunhai Tong},
  journal={arXiv preprint arXiv:2412.07589},
  year={2024},
}
```



<p align="center">
  <a href="https://star-history.com/#jianzongwu/DiffSensei&Date">
    <img src="https://api.star-history.com/svg?repos=jianzongwu/DiffSensei&type=Date" alt="Star History Chart">
  </a>
</p>
