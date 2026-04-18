# Hermes Gemma 4 Colab

自己改善型AIエージェントの個人インフラ(Google Colab版)

## 構成

- **基盤モデル**: Gemma 4 31B (via vLLM, OpenAI互換API)
- **永続記憶**: Google Drive (`/MyDrive/hermes_gemma4/`)
- **データリポジトリ**: [zoffy/hermes-memory](https://huggingface.co/datasets/zoffy/hermes-memory) (HF Hub)
- **コードリポジトリ**: このリポジトリ (GitHub)

## 復活方法(ランタイム切断後)

新しいColabランタイムで次の3行を実行するだけ:

```bash
!git clone https://github.com/token-secure/hermes-gemma4-colab.git
%cd hermes-gemma4-colab
!python scripts/revive.py
```

## ディレクトリ

- `src/` - Pythonクライアント実装
- `configs/` - Hermes Agent設定
- `scripts/` - 復活スクリプト等
- `notebooks/` - 構築ノートブック

## 前提条件

- Colab Pro+(A100 80GB)
- ColabシークレットにHF_TOKEN(Write権限)登録済み
- HuggingFace で google/gemma-4-31B-it のライセンス同意済み

Last updated: 2026-04-18
