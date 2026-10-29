"""Схемы данных для ответов API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class UserGet(BaseModel):
    """Данные пользователя."""

    id: int
    gender: int
    age: int
    country: str
    city: str
    exp_group: int
    os: str
    source: str


class PostGet(BaseModel):
    """Данные поста в списке рекомендаций."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    text: str
    topic: str


class Response(BaseModel):
    """A/B-группа и рекомендованные посты."""

    exp_group: str
    recommendations: list[PostGet]


class FeedGet(BaseModel):
    """Событие взаимодействия пользователя с постом."""

    user_id: int
    post_id: int
    user: UserGet
    post: PostGet
    action: str
    time: datetime
