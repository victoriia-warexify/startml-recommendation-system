"""Проверка когорты и переиспользования SQL без подключения к PostgreSQL."""

import pandas as pd
import pytest

import connection
import ablation_study


class FakeConnection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_default_load_data_uses_five_percent_cohort(monkeypatch):
    fake_conn = FakeConnection()
    queries = []

    def fake_read_sql(query, conn, params=None):
        assert conn is fake_conn
        queries.append((query, params))
        if 'public.user_data' in query:
            return pd.DataFrame({'user_id': [100, 101, 200]})
        if 'public.post_text_df' in query:
            return pd.DataFrame({'post_id': [1, 2]})
        if 'public.feed_data' in query:
            return pd.DataFrame({'user_id': [100, 200], 'post_id': [1, 2]})
        raise AssertionError('Неожиданный SQL')

    monkeypatch.setattr(connection, 'postgres_connection', lambda: fake_conn)
    monkeypatch.setattr(connection.pd, 'read_sql', fake_read_sql)

    users, posts, feed = connection.load_data()
    assert list(users.user_id) == [100, 101, 200]  # все пользователи
    assert list(posts.post_id) == [1, 2]  # все посты
    assert list(feed.user_id) == [100, 200]
    assert 'MOD(user_id, %(user_mod)s) = 0' in queries[2][0]
    assert queries[2][1] == {'user_mod': 20}
    assert "timestamp >= '2021-10-01'" in queries[2][0]
    assert "timestamp < '2021-12-30'" in queries[2][0]
    assert fake_conn.closed


def test_user_mod_parameter_is_bound_not_inserted_into_sql(monkeypatch):
    fake_conn = FakeConnection()
    calls = []

    def fake_read_sql(query, conn, params=None):
        calls.append((query, params))
        return pd.DataFrame()

    monkeypatch.setattr(connection, 'postgres_connection', lambda: fake_conn)
    monkeypatch.setattr(connection.pd, 'read_sql', fake_read_sql)
    connection.load_data(user_mod=50)
    assert calls[2][1] == {'user_mod': 50}
    assert 'user_id % 50' not in calls[2][0]
    assert fake_conn.closed


@pytest.mark.parametrize('invalid', [0, -10, 1.5, '100', True])
def test_rejects_invalid_sample_modulus(monkeypatch, invalid):
    def fail_connection():
        raise AssertionError('При неверном user_mod подключение не должно начинаться')

    monkeypatch.setattr(connection, 'postgres_connection', fail_connection)
    with pytest.raises(ValueError, match='user_mod'):
        connection.load_data(user_mod=invalid)


def test_connection_closes_on_read_error(monkeypatch):
    fake_conn = FakeConnection()
    monkeypatch.setattr(connection, 'postgres_connection', lambda: fake_conn)

    def fail_read(*args, **kwargs):
        raise RuntimeError('SQL error')

    monkeypatch.setattr(connection.pd, 'read_sql', fail_read)
    with pytest.raises(RuntimeError, match='SQL error'):
        connection.load_data()
    assert fake_conn.closed


def test_ablation_reuses_common_load_data(monkeypatch):
    users = pd.DataFrame({'user_id': [50]})
    posts = pd.DataFrame({'post_id': [10]})
    feed = pd.DataFrame({'user_id': [50], 'post_id': [10]})
    calls = []

    def fake_load_data(user_mod=100):
        calls.append(user_mod)
        return users, posts, feed

    monkeypatch.setattr(ablation_study, 'load_data', fake_load_data)
    actual = ablation_study.load_sample(50)
    assert actual[0] is users and actual[1] is posts and actual[2] is feed
    assert calls == [50]
