"""Проверки экспорта признаков без чтения и записи в PostgreSQL."""

from contextlib import AbstractContextManager

import pandas as pd
import pytest

import save_features


def sample_export():
    cutoff = save_features.SPLIT_DATE.isoformat()
    data = {
        "post_features": {"post_id": [1]},
        "user_features": {"user_id": [20]},
        "user_topic_features": {"user_id": [20], "topic": ["news"]},
        "user_cluster_features": {"user_id": [20], "text_cluster": ["news:c00"]},
        "user_seen_features": {"user_id": [20]},
    }
    return {
        name: pd.DataFrame({
            **columns,
            "_feature_definition": [save_features.FEATURE_DEFINITION],
            "_history_cutoff": [cutoff],
        })
        for name, columns in data.items()
    }


@pytest.mark.parametrize("name", list(save_features.TABLE_KEYS))
def test_export_rejects_repeated_keys(name):
    frames = sample_export()
    frames[name] = pd.concat([frames[name], frames[name]], ignore_index=True)
    with pytest.raises(ValueError, match="повторяющиеся ключи"):
        save_features.validate_export_tables(frames)


def test_export_rejects_wrong_metadata():
    frames = sample_export()
    frames["post_features"].loc[0, "_history_cutoff"] = "2021-01-01T00:00:00"
    with pytest.raises(ValueError, match="дата истории"):
        save_features.validate_export_tables(frames)
    frames = sample_export()
    frames["user_features"].loc[0, "_feature_definition"] = "old_features"
    with pytest.raises(ValueError, match="версия признаков"):
        save_features.validate_export_tables(frames)


def test_export_rejects_null_keys_and_missing_tables():
    frames = sample_export()
    frames["user_topic_features"].loc[0, "topic"] = None
    with pytest.raises(ValueError, match="пустые или повторяющиеся"):
        save_features.validate_export_tables(frames)
    frames = sample_export()
    del frames["user_seen_features"]
    with pytest.raises(ValueError, match="Отсутствует таблица"):
        save_features.validate_export_tables(frames)


class FakeTransaction(AbstractContextManager):
    def __init__(self):
        self.failed = False
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, typ, value, tb):
        self.exited = True
        self.failed = typ is not None
        return False


class FakeEngine:
    def __init__(self):
        self.transaction = FakeTransaction()
        self.disposed = False

    def begin(self):
        return self.transaction

    def dispose(self):
        self.disposed = True


@pytest.fixture
def export_setup(monkeypatch):
    monkeypatch.setenv("FEATURE_TABLE_PREFIX", "recsys_test")
    calls = {"loads": [], "sql": []}
    engine = FakeEngine()

    def fake_load_data(*, user_mod):
        calls["loads"].append(user_mod)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(save_features, "load_data", fake_load_data)
    monkeypatch.setattr(save_features, "temporal_split_masks", lambda feed: None)
    monkeypatch.setattr(save_features, "build_post_features", lambda posts, fit_pca: pd.DataFrame())
    monkeypatch.setattr(save_features, "build_history_snapshot", lambda *args: {})
    monkeypatch.setattr(save_features, "snapshot_export_tables", lambda snapshot: sample_export())
    monkeypatch.setattr(save_features, "create_db_engine", lambda: engine)

    class Inspector:
        def has_table(self, name, schema):
            return False

    monkeypatch.setattr(save_features, "inspect", lambda conn: Inspector())

    def fake_to_sql(self, name, *, con, schema, if_exists, index, chunksize, method):
        assert con is engine.transaction
        calls["sql"].append((name, schema, if_exists, index, chunksize, method))

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql)
    return engine, calls, monkeypatch


def test_export_requires_explicit_write_permission(monkeypatch):
    monkeypatch.setattr(save_features, "load_data", lambda **kwargs: pytest.fail("БД не должна читаться"))
    with pytest.raises(SystemExit):
        save_features.main([])


def test_export_rejects_invalid_user_mod_and_prefix(monkeypatch):
    monkeypatch.setenv("FEATURE_TABLE_PREFIX", "recsys_test")
    monkeypatch.setattr(save_features, "load_data", lambda **kwargs: pytest.fail("БД не должна читаться"))
    with pytest.raises(SystemExit):
        save_features.main(["--allow-db-write", "--user-mod", "0"])
    monkeypatch.setenv("FEATURE_TABLE_PREFIX", "bad prefix")
    with pytest.raises(RuntimeError, match="FEATURE_TABLE_PREFIX"):
        save_features.main(["--allow-db-write"])


def test_export_passes_cohort_and_creates_five_new_tables(export_setup):
    engine, calls, _ = export_setup
    save_features.main(["--allow-db-write", "--user-mod", "50"])
    assert calls["loads"] == [50]
    assert [call[0] for call in calls["sql"]] == [
        "recsys_test_" + key for key in save_features.TABLE_KEYS
    ]
    assert all(call[1:4] == ("public", "fail", False) for call in calls["sql"])
    assert engine.transaction.exited and not engine.transaction.failed
    assert engine.disposed


def test_existing_table_prevents_any_write(export_setup):
    engine, calls, monkeypatch = export_setup

    class Inspector:
        def has_table(self, name, schema):
            return name == "recsys_test_post_features"

    monkeypatch.setattr(save_features, "inspect", lambda conn: Inspector())
    with pytest.raises(RuntimeError, match="перезапись запрещена"):
        save_features.main(["--allow-db-write"])
    assert calls["sql"] == []
    assert engine.transaction.failed
    assert engine.disposed


def test_export_error_leaves_transaction_to_rollback(export_setup):
    engine, calls, monkeypatch = export_setup

    def broken_to_sql(self, name, **kwargs):
        calls["sql"].append(name)
        if name.endswith("user_topic_features"):
            raise RuntimeError("Имитация ошибки INSERT")

    monkeypatch.setattr(pd.DataFrame, "to_sql", broken_to_sql)
    with pytest.raises(RuntimeError, match="INSERT"):
        save_features.main(["--allow-db-write"])
    assert len(calls["sql"]) == 3
    assert engine.transaction.failed
    assert engine.disposed
