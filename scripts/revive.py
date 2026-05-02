#!/usr/bin/env python
"""
Hermes Gemma 4 - 復活スクリプト v2
=====================================
ランタイムが切れた後、Colab セル内から呼び出して環境を復元する。

【v1 からの変更点】
- subprocess から `google.colab.drive.mount` / `userdata.get` を呼ぶ
  バグを修正。これらの Colab API は呼び出し元の Colab セル側で
  事前に実行しておく前提に変更（HF_TOKEN は環境変数で受け取る）。
- 責務を「依存インストール + vLLM 起動 + ヘルスチェック」に集中。
- A100 80GB チェックを内部に追加（40GB を引いた場合の早期検知）。

【使い方】(Colab セル内)
    # === 事前にセル側で実行する前提 ===
    from google.colab import drive, userdata
    import os
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    drive.mount("/content/drive")

    # === revive を呼ぶ ===
    import sys
    sys.path.insert(0, "scripts")
    from revive import main
    main()
"""
import os
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: str, check: bool = True):
    """シェルコマンド実行。stdout はそのまま流す。"""
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def check_gpu():
    """A100 80GB が引けているかをチェック。40GB だと OOM するので早期失敗。"""
    print("\n[1/4] GPU チェック...")
    try:
        import torch
    except ImportError:
        print("  ⚠️ torch 未インストール（依存インストール後に再チェック）")
        return None

    if not torch.cuda.is_available():
        print("❌ CUDA が使えません。ランタイムタイプを GPU に変更してください。")
        sys.exit(1)

    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  Device: {name}")
    print(f"  VRAM:   {vram_gb:.1f} GB")

    if vram_gb < 70:
        print(f"\n❌ VRAM 不足: {vram_gb:.1f} GB（Gemma 4 31B には 80GB が必要）")
        print("   👉 ランタイムを「接続解除して削除」→ A100 を再接続してください。")
        print("   👉 Colab Pro+ でも 40GB / 80GB はガチャです。")
        sys.exit(1)
    print(f"  ✅ A100 80GB 確認")


def install_deps():
    """vLLM + transformers 等の依存をインストール。"""
    print("\n[2/4] 依存パッケージ インストール...")
    run(
        "pip install -q "
        "'transformers>=4.57.0' 'vllm>=0.11.0' "
        "'tokenizers>=0.21' 'accelerate>=0.34' "
        "huggingface_hub pyyaml openai"
    )
    print("  ✅ インストール完了")


def check_env():
    """必須の環境変数（HF_TOKEN, Drive マウント）を確認。"""
    print("\n[3/4] 環境変数 / Drive マウント確認...")

    # HF_TOKEN
    if not os.environ.get("HF_TOKEN"):
        print("\n❌ HF_TOKEN が環境変数に設定されていません。")
        print("   Colab セル側で以下を実行してから revive を呼んでください:")
        print("     from google.colab import userdata")
        print("     import os")
        print("     os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')")
        print("     os.environ['HUGGING_FACE_HUB_TOKEN'] = os.environ['HF_TOKEN']")
        sys.exit(1)
    print(f"  ✅ HF_TOKEN: {os.environ['HF_TOKEN'][:7]}...{os.environ['HF_TOKEN'][-4:]}")

    # Drive
    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.exists():
        print("\n❌ Google Drive がマウントされていません。")
        print("   Colab セル側で以下を実行してから revive を呼んでください:")
        print("     from google.colab import drive")
        print("     drive.mount('/content/drive')")
        sys.exit(1)
    print(f"  ✅ Drive マウント済み: {drive_root}")


def start_vllm(timeout_sec: int = 600):
    """vLLM を nohup でバックグラウンド起動し、/health でポーリング。"""
    print("\n[4/4] vLLM + Gemma 4 31B 起動 (5〜10 分)...")

    # 既存プロセスを完全終了
    run("pkill -9 -f vllm || true", check=False)
    time.sleep(3)

    Path("logs").mkdir(exist_ok=True)

    # A100 80GB 用に最適化された起動コマンド
    # 注意: vLLM 0.20+ では:
    #   - CUDA Graph profiling がデフォルトONになり KV cache 使用可能量が減る
    #   - Gemma 4 のマルチモーダル制約で --max-num-batched-tokens を明示的に指定する必要がある
    # 対応: gpu-memory-utilization=0.95, max-model-len=10240 で安定動作確認済み
    cmd = (
        "nohup vllm serve google/gemma-4-31B-it "
        "--host 0.0.0.0 --port 8000 "
        "--dtype bfloat16 "
        "--gpu-memory-utilization 0.95 "
        "--max-model-len 10240 "
        "--max-num-seqs 16 "
        "--max-num-batched-tokens 10240 "
        "--enable-prefix-caching "
        "> logs/vllm.log 2>&1 &"
    )
    os.system(cmd)
    print("  🚀 起動コマンド発行（バックグラウンド）")

    # ヘルスチェック
    import requests
    start = time.time()
    success = False
    while time.time() - start < timeout_sec:
        elapsed = int(time.time() - start)
        try:
            r = requests.get("http://localhost:8000/health", timeout=5)
            if r.status_code == 200:
                print(f"\n  ✅ 起動成功（所要: {elapsed} 秒）")
                success = True
                break
        except requests.exceptions.RequestException:
            pass

        if elapsed % 30 == 0 and elapsed > 0:
            print(f"  ⏳ 待機中... {elapsed} 秒経過 / 最大 {timeout_sec} 秒")
            # ログの直近を要約表示
            try:
                lines = Path("logs/vllm.log").read_text().splitlines()
                for kw in ["Loaded", "ERROR", "Traceback", "started"]:
                    for line in reversed(lines[-50:]):
                        if kw.lower() in line.lower():
                            print(f"     └─ {line.strip()[:100]}")
                            break
            except Exception:
                pass
        time.sleep(5)

    if not success:
        print(f"\n❌ {timeout_sec} 秒以内に起動しませんでした。logs/vllm.log を確認してください。")
        run("tail -60 logs/vllm.log", check=False)
        sys.exit(1)


def main():
    print("=" * 60)
    print("  Hermes Gemma 4 Revival Script v2")
    print("=" * 60)

    check_env()       # [3/4] を先に行う（依存インストール前のチェック）
    install_deps()    # [2/4]
    check_gpu()       # [1/4] torch 入れてからじゃないと判定不能
    start_vllm()      # [4/4]

    print("\n" + "=" * 60)
    print("  ✅ Hermes Gemma 4 復活完了")
    print("=" * 60)
    print("\n次のセルで v3 クライアントを起動してください:")
    print("  import sys")
    print("  sys.path.insert(0, 'src')")
    print("  from hermes_client import HermesGemma4Client")
    print('  client = HermesGemma4Client(instance="03_intelligence")')
    print('  print(client.chat("こんにちは"))')


if __name__ == "__main__":
    # スクリプト直接実行は非推奨だが、後方互換のため残す
    print("⚠️ スクリプト直接実行は非推奨です。Colab セル内から import してください。")
    print("   詳しくはファイル冒頭の docstring を参照してください。\n")
    main()
