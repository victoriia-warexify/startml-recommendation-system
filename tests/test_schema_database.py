"""Проверка схем API и имён таблиц без подключения к PostgreSQL."""

import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from database import feature_table
from schema import PostGet, Response


def test_post_from_object():
    post = SimpleNamespace(id=42, text="Пост", topic="science")
    result = PostGet.model_validate(post)
    assert result.model_dump() == {"id": 42, "text": "Пост", "topic": "science"}


def test_response_serialization():
    response = Response(exp_group="test", recommendations=[
        {"id": 1, "text": "Первый", "topic": "tech"},
        {"id": 2, "text": "Второй", "topic": "news"},
    ])
    assert response.exp_group == "test"
    assert [post.id for post in response.recommendations] == [1, 2]
    assert response.model_dump()["recommendations"][0]["text"] == "Первый"


def test_missing_post_fields_are_rejected():
    with pytest.raises(ValidationError):
        PostGet.model_validate({"id": 1, "topic": "tech"})


def test_feature_table_uses_prefix_and_whitelist(monkeypatch):
    monkeypatch.setenv("FEATURE_TABLE_PREFIX", "recsys_5pct")
    assert feature_table("post_features") == "recsys_5pct_post_features"
    assert feature_table("user_seen_features") == "recsys_5pct_user_seen_features"
    with pytest.raises(ValueError, match="Неизвестный тип"):
        feature_table("feed_data")


@pytest.mark.parametrize("prefix", ["", "bad prefix", 'x";DROP TABLE users;--'])
def test_feature_table_rejects_bad_prefix(monkeypatch, prefix):
    monkeypatch.setenv("FEATURE_TABLE_PREFIX", prefix)
    with pytest.raises(RuntimeError, match="FEATURE_TABLE_PREFIX"):
        feature_table("post_features")



def test_database_engine_configuration_without_connection(monkeypatch):
    import database

    for key, value in {
        "PGHOST": "localhost", "PGPORT": "5432",
        "PGDATABASE": "example", "PGUSER": "demo", "PGPASSWORD": "fake-secret",
    }.items():
        monkeypatch.setenv(key, value)

    captured = {}

    def fake_engine(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return "fake engine"

    monkeypatch.setattr(database, "create_engine", fake_engine)
    assert database.create_db_engine() == "fake engine"
    assert captured["url"].host == "localhost"
    assert captured["url"].database == "example"
    assert captured["url"].username == "demo"
    assert captured["url"].password == "fake-secret"
    assert captured["kwargs"]["hide_parameters"] is True


def test_postgres_connection_enables_autocommit(monkeypatch):
    from database import postgres_connection

    for key, value in {
        "PGHOST": "localhost", "PGPORT": "5432",
        "PGDATABASE": "example", "PGUSER": "demo", "PGPASSWORD": "fake-secret",
    }.items():
        monkeypatch.setenv(key, value)

    captured = {}

    class FakeConnection:
        autocommit = False

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return FakeConnection()

    monkeypatch.setitem(sys.modules, "psycopg2", SimpleNamespace(connect=fake_connect))
    connection = postgres_connection()
    assert connection.autocommit is True
    assert captured["database"] == "example"
    assert captured["password"] == "fake-secret"
