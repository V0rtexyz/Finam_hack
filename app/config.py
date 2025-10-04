from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    dataset_dir: Path = project_root / "dataset"
    dataset_file: Path = dataset_dir / "result.json"
    artifacts_dir: Path = project_root / "artifacts"
    reports_dir: Path = project_root / "reports"

    # Models
    embedding_model: str = "text-embedding-3-small"

    # Topic modeling
    default_max_clusters: int = 30
    default_min_clusters: int = 5

    # Caching
    embeddings_sqlite: Path = artifacts_dir / "embeddings_cache.sqlite"
    messages_parquet: Path = artifacts_dir / "messages.parquet"
    projections_parquet: Path = artifacts_dir / "projections.parquet"
    clusters_parquet: Path = artifacts_dir / "clusters.parquet"
    cluster_labels_json: Path = artifacts_dir / "cluster_labels.json"

    # Telegram integration
    telegram_session_file: Path = artifacts_dir / "telegram.session"
    telegram_channels: tuple[str, ...] = (
        "tb_invest_official",
        "SberInvestments",
        "alfa_investments",
        "centralbank_russia",
    )
    telegram_max_messages: int = 3000
    telegram_phone: str = os.getenv("TELEGRAM_PHONE", "+79638900693")

    @property
    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    @property
    def openai_base_url(self) -> str | None:
        return os.getenv("OPENAI_BASE_URL")

    @property
    def yandex_folder_id(self) -> str | None:
        return os.getenv("FOLDER_ID")

    @property
    def yandex_service_account_id(self) -> str | None:
        return os.getenv("SERVICE_ACCOUNT_ID")

    @property
    def yandex_key_id(self) -> str | None:
        return os.getenv("KEY_ID")

    @property
    def yandex_private_key_path(self) -> str | None:
        return os.getenv("PRIVATE_KEY")

    @property
    def has_yandex_credentials(self) -> bool:
        return all(
            [
                self.yandex_folder_id,
                self.yandex_service_account_id,
                self.yandex_key_id,
                self.yandex_private_key_path,
            ]
        )

    @property
    def chat_model(self) -> str:
        return os.getenv("CHAT_MODEL", "yandexgpt")

    @property
    def telegram_api_id(self) -> int | None:
        value = os.getenv("TELEGRAM_API_ID") or os.getenv("TG_API_ID") or "25373049"
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @property
    def telegram_api_hash(self) -> str | None:
        return os.getenv("TELEGRAM_API_HASH") or os.getenv("TG_API_HASH") or "a3d20525369442418dd19441df6c8c6f"

    @property
    def telegram_session_string(self) -> str | None:
        return os.getenv("TELEGRAM_SESSION_STRING")

    @property
    def telegram_session_path(self) -> Path:
        env_value = os.getenv("TELEGRAM_SESSION_PATH") or os.getenv("TG_SESSION_PATH")
        if env_value:
            return Path(env_value).expanduser()
        return self.telegram_session_file

    @property
    def has_telegram_credentials(self) -> bool:
        return bool(self.telegram_api_id and self.telegram_api_hash)


cfg = AppConfig()

# Ensure directories exist at import-time for convenience
cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
cfg.reports_dir.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env if present
load_dotenv()


