import pandas as pd

from database import postgres_connection


def load_data(user_mod: int = 20):
    """Загружает пользователей, посты и историю выбранной когорты.

    История ограничена user_id % user_mod == 0: 20 — около 5%,
    50 — около 2%, 100 — около 1%. Справочники загружаются целиком.
    """
    if not isinstance(user_mod, int) or isinstance(user_mod, bool) or user_mod < 1:
        raise ValueError("user_mod должен быть положительным целым числом")
    conn = postgres_connection()

    try:
        users_df = pd.read_sql(
            """
            SELECT *
            FROM public.user_data
            """,
            conn
        )

        posts_df = pd.read_sql(
            """
            SELECT *
            FROM public.post_text_df
            """,
            conn
        )

        feed_df = pd.read_sql(
            """
            SELECT user_id, post_id, action, timestamp, target
            FROM public.feed_data
            WHERE timestamp >= '2021-10-01'
            AND timestamp < '2021-12-30'
            AND MOD(user_id, %(user_mod)s) = 0
            """,
            conn,
            params={"user_mod": user_mod}
        )

    finally:
        conn.close()

    return users_df, posts_df, feed_df
