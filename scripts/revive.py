#!/usr/bin/env python
"""
Hermes Gemma 4 - 復活スクリプト
ランタイムが切れた後、このスクリプト1つで全環境を復元する

使い方(Colabの新しいランタイムで):
    !git clone https://github.com/token-secure/hermes-gemma4-colab.git
    %cd hermes-gemma4-colab
    !python scripts/revive.py
"""
import subprocess, sys, os, time
from pathlib import Path

def run(cmd, shell=True):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=shell, check=True)

def main():
    print("=" * 60)
    print("Hermes Gemma 4 Revival Script")
    print("=" * 60)

    # 1. 依存インストール
    print("\n[1/5] 依存パッケージ インストール中...")
    run("pip install -q 'transformers>=4.57.0' 'vllm>=0.11.0' "
        "'tokenizers>=0.21' 'accelerate>=0.34' huggingface_hub pyyaml")

    # 2. Drive マウント
    print("\n[2/5] Google Drive マウント中...")
    from google.colab import drive
    drive.mount("/content/drive")

    # 3. HF_TOKEN 確認
    print("\n[3/5] HF_TOKEN 確認...")
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    # 4. vLLM 起動
    print("\n[4/5] vLLM + Gemma 4 31B 起動中(5-10分)...")
    Path("logs").mkdir(exist_ok=True)
    run("pkill -9 -f vllm || true")
    time.sleep(3)
    run("""nohup vllm serve google/gemma-4-31B-it \\
        --host 0.0.0.0 --port 8000 \\
        --dtype bfloat16 --gpu-memory-utilization 0.90 \\
        --max-model-len 16384 --max-num-seqs 16 \\
        --enable-prefix-caching \\
        > logs/vllm.log 2>&1 &""")

    # 5. 起動待機
    print("\n[5/5] サーバー起動待機中...")
    import requests
    for i in range(120):
        try:
            r = requests.get("http://localhost:8000/health", timeout=3)
            if r.status_code == 200:
                print(f"\n✅ 起動成功(所要: {i*5}秒)")
                break
        except:
            pass
        if i % 6 == 0:
            print(f"  {i*5}秒経過...")
        time.sleep(5)
    else:
        print("❌ タイムアウト")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ Hermes Gemma 4 復活完了")
    print("=" * 60)
    print("\n使用例:")
    print("  sys.path.insert(0, 'src')")
    print("  from hermes_client import HermesGemma4Client")
    print("  client = HermesGemma4Client()")
    print("  print(client.chat('こんにちは'))")

if __name__ == "__main__":
    main()
