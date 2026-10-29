"""Экспорт исторических признаков в новые таблицы PostgreSQL."""

import argparse
import os

import pandas as pd
from sqlalchemy import inspect

from connection import load_data
from database import create_db_engine, feature_table
from features import (
    FEATURE_DEFINITION,
    SPLIT_DATE,
    build_history_snapshot,
    build_post_features,
    snapshot_export_tables,
    temporal_split_masks,
)

TABLE_KEYS = {
    "post_features": ["post_id"],
    "user_features": ["user_id"],
    "user_topic_features": ["user_id", "topic"],
    "user_cluster_features": ["user_id", "text_cluster"],
    "user_seen_features": ["user_id"],
}
TABLE_METADATA = ("_feature_definition", "_history_cutoff")


def validate_export_tables(frames: dict[str, pd.DataFrame]) -> None:
    """Проверяет ключи и метаданные таблиц до обращения к PostgreSQL."""
    expected_cutoff = SPLIT_DATE.isoformat()

    for name, key in TABLE_KEYS.items():
        if name not in frames:
            raise ValueError(f"Отсутствует таблица {name}")
        frame = frames[name]
        required = [*key, *TABLE_METADATA]
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise ValueError(f"В {name} нет столбцов: {missing}")
        if frame[key].isna().any().any() or frame.duplicated(key).any():
            raise ValueError(f"В {name} пустые или повторяющиеся ключи: {key}")
        if not frame.empty:
            if not frame["_feature_definition"].eq(FEATURE_DEFINITION).all():
                raise ValueError(f"В {name} неверная версия признаков")
            if not frame["_history_cutoff"].eq(expected_cutoff).all():
                raise ValueError(f"В {name} неверная дата истории")


def main(argv: list[str] | None = None) -> None:
    """Создаёт пять таблиц признаков, не перезаписывая существующие."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-db-write", action="store_true")
    parser.add_argument(
        "--user-mod", type=int, default=20,
        help="Модуль отбора пользователей: 20 — около 5%%, 50 — около 2%%",
    )
    args = parser.parse_args(argv)

    if not args.allow_db_write:
        parser.error("Для записи в PostgreSQL требуется --allow-db-write")
    if args.user_mod < 1:
        parser.error("--user-mod должен быть положительным")
    if not os.environ.get("FEATURE_TABLE_PREFIX"):
        parser.error("Укажите FEATURE_TABLE_PREFIX для новых таблиц")

    # Проверяем имена таблиц до загрузки данных и создания подключения для записи.
    names = {kind: feature_table(kind) for kind in TABLE_KEYS}
    print(f"Когорта: user_id % {args.user_mod} = 0", flush=True)
    users, posts, feed = load_data(user_mod=args.user_mod)
    temporal_split_masks(feed)
    print(f"users: {users.shape}; posts: {posts.shape}; feed: {feed.shape}")

    base_posts = build_post_features(posts, fit_pca=False)
    snapshot = build_history_snapshot(users, base_posts, feed)
    frames = snapshot_export_tables(snapshot)
    validate_export_tables(frames)

    for name, frame in frames.items():
        print(f"{name}: {frame.shape}")

    engine = create_db_engine()
    try:
        # Транзакция откатит все пять записей при любой ошибке.
        with engine.begin() as conn:
            inspector = inspect(conn)
            existing = [name for name in names.values() if inspector.has_table(name, schema="public")]
            if existing:
                raise RuntimeError(f"Таблицы уже существуют, перезапись запрещена: {existing}")

            for kind, name in names.items():
                frames[kind].to_sql(
                    name, con=conn, schema="public", if_exists="fail",
                    index=False, chunksize=500 if kind in ("post_features", "user_features") else 1000,
                    method="multi",
                )
    finally:
        engine.dispose()

    print("Все пять таблиц признаков сохранены")


if __name__ == "__main__":
    main()
