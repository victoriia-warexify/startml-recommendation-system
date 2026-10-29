import os
import re
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL


# .env лежит рядом с database.py; уже заданные переменные окружения не заменяем.
load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)


def postgres_connection():
    """Устанавливает и возвращает соединение с PostgreSQL."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.environ["PGHOST"],
            port=int(os.getenv("PGPORT", "5432")),
            database=os.environ["PGDATABASE"],
            user=os.environ["PGUSER"],
            password=os.environ["PGPASSWORD"],
        )
    except (KeyError, ValueError):
        raise RuntimeError("Проверьте PGHOST, PGPORT, PGDATABASE, PGUSER и PGPASSWORD в .env") from None
    except Exception:
        # Не показываем текст ошибки драйвера: он может содержать реквизиты.
        raise RuntimeError("Ошибка при подключении к PostgreSQL") from None

    conn.autocommit = True

    return conn


def create_db_engine():
    """SQLAlchemy Engine для save_features.py, с теми же настройками .env."""
    try:
        url = URL.create(
            "postgresql+psycopg2",
            username=os.environ["PGUSER"],
            password=os.environ["PGPASSWORD"],
            host=os.environ["PGHOST"],
            port=int(os.getenv("PGPORT", "5432")),
            database=os.environ["PGDATABASE"],
        )
        return create_engine(url, pool_pre_ping=True, hide_parameters=True)
    except (KeyError, ValueError):
        raise RuntimeError("Проверьте параметры PostgreSQL в .env") from None


def feature_table(kind: str) -> str:
    """Имя одной из пяти таблиц признаков для app.py/save_features.py."""
    allowed = {
        "post_features",
        "user_features",
        "user_topic_features",
        "user_cluster_features",
        "user_seen_features",
    }
    if kind not in allowed:
        raise ValueError(f"Неизвестный тип таблицы: {kind}")

    prefix = os.getenv("FEATURE_TABLE_PREFIX", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
        raise RuntimeError("Укажите корректный FEATURE_TABLE_PREFIX в .env")

    return f"{prefix}_{kind}"
