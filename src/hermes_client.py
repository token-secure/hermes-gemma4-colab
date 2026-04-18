"""
Hermes Gemma 4 Client v2
プロファイル自動読み込み + 永続化対話
"""
import json
from pathlib import Path
from datetime import datetime
from openai import OpenAI

DRIVE_DIR = Path("/content/drive/MyDrive/hermes_gemma4")

class HermesGemma4Client:
    def __init__(self, base_url="http://localhost:8000/v1", load_profile=True):
        self.client = OpenAI(base_url=base_url, api_key="dummy")
        self.model = "google/gemma-4-31B-it"
        self.conversation = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.profile = self._load_profile() if load_profile else {}

    def _load_profile(self) -> dict:
        """Driveからユーザープロファイルを読み込む"""
        profile_path = DRIVE_DIR / "user_profile.json"
        if profile_path.exists():
            return json.loads(profile_path.read_text())
        return {}

    def _build_system_prompt(self, custom_system: str = None) -> str:
        """プロファイルを組み込んだシステムプロンプトを生成"""
        base = custom_system or "あなたは永続記憶を持つ自己改善型AIアシスタントです。"
        if not self.profile:
            return base

        profile_str = "\n\n## ユーザープロファイル\n"
        if "hf_username" in self.profile:
            profile_str += f"- HuggingFace: {self.profile['hf_username']}\n"
        if "github_username" in self.profile:
            profile_str += f"- GitHub: {self.profile['github_username']}\n"
        if "interests" in self.profile:
            interests = "、".join(self.profile["interests"])
            profile_str += f"- 興味・関心: {interests}\n"
        if "languages" in self.profile:
            profile_str += f"- 使用言語: {', '.join(self.profile['languages'])}\n"

        profile_str += "\nこのプロファイルを踏まえて、自然に会話してください。"
        return base + profile_str

    def chat(self, message: str, system: str = None, save: bool = True) -> str:
        """対話1ターンを実行、Driveに記録"""
        if not self.conversation:
            full_system = self._build_system_prompt(system)
            self.conversation.append({"role": "system", "content": full_system})

        self.conversation.append({"role": "user", "content": message})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation,
            temperature=0.7,
            max_tokens=1024,
        )
        reply = response.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": reply})
        if save:
            self._save_turn()
        return reply

    def _save_turn(self):
        """対話ログをDriveに追記保存"""
        log_dir = DRIVE_DIR / "conversations"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{self.session_id}.jsonl"
        latest_turn = {
            "timestamp": datetime.now().isoformat(),
            "user": self.conversation[-2]["content"],
            "assistant": self.conversation[-1]["content"],
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(latest_turn, ensure_ascii=False) + "\n")

    def update_profile(self, updates: dict):
        """プロファイルを更新してDriveに保存"""
        self.profile.update(updates)
        profile_path = DRIVE_DIR / "user_profile.json"
        profile_path.write_text(json.dumps(self.profile, ensure_ascii=False, indent=2))

    def reset(self):
        """新しいセッション開始(プロファイルは維持)"""
        self.conversation = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
