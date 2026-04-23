"""
Hermes Gemma 4 Client v3
- マルチインスタンス対応
- INSTANCE名で完全に分離された記憶領域
- プロファイル自動読み込み + 永続化対話
"""
import json
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# ==================== インスタンス定義 ====================
BASE_DRIVE = Path("/content/drive/MyDrive")

INSTANCE_CONFIG = {
    "01_arxiv":        {"hf_repo": "zoffy/hermes-memory-arxiv",        "role": "arXiv論文読解・知識DB"},
    "02_avatar":       {"hf_repo": "zoffy/hermes-memory-avatar",       "role": "3Dアバター生成(SentiAvatar)"},
    "03_intelligence": {"hf_repo": "zoffy/hermes-memory-intelligence", "role": "パーソナルインテリジェンス"},
    "04_blender":      {"hf_repo": "zoffy/hermes-memory-blender",      "role": "Blender AI統合"},
    "05_cdpa":         {"hf_repo": "zoffy/hermes-memory-cdpa",         "role": "CDPA推論エンジン"},
}


class HermesGemma4Client:
    def __init__(self, instance: str, base_url: str = "http://localhost:8000/v1", load_profile: bool = True):
        """
        Args:
            instance: INSTANCE_CONFIG のキー(例: "03_intelligence")
            base_url: vLLM サーバーのエンドポイント
            load_profile: 起動時にプロファイルを自動読み込むか
        """
        # インスタンス検証
        if instance not in INSTANCE_CONFIG:
            valid = ", ".join(INSTANCE_CONFIG.keys())
            raise ValueError(
                f"❌ 不明なインスタンス: '{instance}'\n"
                f"   有効な値: {valid}"
            )
        
        self.instance = instance
        self.config = INSTANCE_CONFIG[instance]
        self.drive_dir = BASE_DRIVE / f"hermes_gemma4_{instance}"
        
        # ディレクトリ自動作成(5インスタンス展開時の初回起動対応)
        self.drive_dir.mkdir(parents=True, exist_ok=True)
        for sub in ["conversations", "skills", "checkpoints", "configs", "memory"]:
            (self.drive_dir / sub).mkdir(exist_ok=True)
        
        # vLLM クライアント
        self.client = OpenAI(base_url=base_url, api_key="dummy")
        self.model = "google/gemma-4-31B-it"
        self.conversation = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.profile = self._load_profile() if load_profile else {}
        
        # 起動時情報表示
        print(f"🎯 Instance: {instance} ({self.config['role']})")
        print(f"📁 Drive: {self.drive_dir}")
        print(f"🤗 HF Repo: {self.config['hf_repo']}")

    def _load_profile(self) -> dict:
        """Driveからユーザープロファイルを読み込む"""
        profile_path = self.drive_dir / "user_profile.json"
        if profile_path.exists():
            return json.loads(profile_path.read_text())
        return {}

    def _build_system_prompt(self, custom_system: str = None) -> str:
        """プロファイル + インスタンス情報を組み込んだシステムプロンプトを生成"""
        base = custom_system or "あなたは永続記憶を持つ自己改善型AIアシスタントです。"
        
        # インスタンスの役割を冒頭に追加
        role_info = f"\n\n## インスタンス情報\n- 専門領域: {self.config['role']}\n"
        
        if not self.profile:
            return base + role_info
        
        profile_str = "\n## ユーザープロファイル\n"
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
        return base + role_info + profile_str

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
        log_dir = self.drive_dir / "conversations"
        log_file = log_dir / f"{self.session_id}.jsonl"
        latest_turn = {
            "timestamp": datetime.now().isoformat(),
            "instance": self.instance,
            "user": self.conversation[-2]["content"],
            "assistant": self.conversation[-1]["content"],
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(latest_turn, ensure_ascii=False) + "\n")

    def update_profile(self, updates: dict):
        """プロファイルを更新してDriveに保存"""
        self.profile.update(updates)
        profile_path = self.drive_dir / "user_profile.json"
        profile_path.write_text(json.dumps(self.profile, ensure_ascii=False, indent=2))

    def reset(self):
        """新しいセッション開始(プロファイルは維持)"""
        self.conversation = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
