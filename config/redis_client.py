from __future__ import annotations

import atexit
import threading
import time

import redis as sync_redis
import redis.asyncio as async_redis
from django.conf import settings

_sync_client = None
_async_client = None


class _MemoryRedisState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.values: dict[str, str | bytes] = {}
        self.expiries: dict[str, float] = {}

    def purge_expired(self, key: str | None = None) -> None:
        now = time.monotonic()
        keys = [key] if key is not None else list(self.expiries)
        for candidate in keys:
            expires_at = self.expiries.get(candidate)
            if expires_at is not None and expires_at <= now:
                self.values.pop(candidate, None)
                self.expiries.pop(candidate, None)


_memory_state = _MemoryRedisState()


class _MemorySyncRedis:
    def __init__(self, state: _MemoryRedisState) -> None:
        self._state = state

    def get(self, key: str):
        with self._state.lock:
            self._state.purge_expired(key)
            value = self._state.values.get(key)
            if value is None:
                return None
            return value if isinstance(value, bytes) else str(value).encode()

    def set(self, key: str, value, nx: bool = False, ex: int | None = None):
        with self._state.lock:
            self._state.purge_expired(key)
            if nx and key in self._state.values:
                return False
            self._state.values[key] = value
            if ex is not None:
                self._state.expiries[key] = time.monotonic() + ex
            else:
                self._state.expiries.pop(key, None)
            return True

    def exists(self, key: str) -> int:
        with self._state.lock:
            self._state.purge_expired(key)
            return int(key in self._state.values)

    def delete(self, *keys: str) -> int:
        deleted = 0
        with self._state.lock:
            for key in keys:
                self._state.purge_expired(key)
                if key in self._state.values:
                    deleted += 1
                    self._state.values.pop(key, None)
                self._state.expiries.pop(key, None)
        return deleted

    def append(self, key: str, value) -> int:
        with self._state.lock:
            self._state.purge_expired(key)
            existing = self._state.values.get(key, b"")
            if isinstance(existing, str):
                existing_bytes = existing.encode()
            else:
                existing_bytes = bytes(existing)
            if isinstance(value, bytes):
                appended = value
            else:
                appended = str(value).encode()
            combined = existing_bytes + appended
            self._state.values[key] = combined
            return len(combined)

    def incr(self, key: str) -> int:
        with self._state.lock:
            self._state.purge_expired(key)
            value = int(self._state.values.get(key, 0)) + 1
            self._state.values[key] = str(value)
            return value

    def decr(self, key: str) -> int:
        with self._state.lock:
            self._state.purge_expired(key)
            value = int(self._state.values.get(key, 0)) - 1
            self._state.values[key] = str(value)
            return value

    def expire(self, key: str, ttl: int) -> int:
        with self._state.lock:
            self._state.purge_expired(key)
            if key not in self._state.values:
                return 0
            self._state.expiries[key] = time.monotonic() + ttl
            return 1

    def eval(self, script: str, numkeys: int, key: str, *args):
        with self._state.lock:
            self._state.purge_expired(key)
            if "redis.call('incr'" in script:
                ttl = int(args[0])
                max_concurrent = int(args[1]) if len(args) > 1 else None
                count = int(self._state.values.get(key, 0)) + 1
                self._state.values[key] = str(count)
                if count == 1:
                    self._state.expiries[key] = time.monotonic() + ttl
                if max_concurrent is None or count <= max_concurrent:
                    return count
                count -= 1
                if count <= 0:
                    self._state.values.pop(key, None)
                    self._state.expiries.pop(key, None)
                else:
                    self._state.values[key] = str(count)
                return 0
            if "redis.call('decr'" in script:
                count = int(self._state.values.get(key, 0)) - 1
                if count <= 0:
                    self._state.values.pop(key, None)
                    self._state.expiries.pop(key, None)
                    return 0
                self._state.values[key] = str(count)
                return count
            if "redis.call('get', KEYS[1]) == ARGV[1]" in script and "expire" in script:
                owner_token = args[0]
                ttl = int(args[1])
                current = self._state.values.get(key)
                if current == owner_token:
                    self._state.expiries[key] = time.monotonic() + ttl
                    return 1
                return 0
            if "redis.call('get', KEYS[1]) == ARGV[1]" in script and "del" in script:
                owner_token = args[0] if args else None
                current = self._state.values.get(key)
                if owner_token is None or current == owner_token:
                    existed = int(key in self._state.values)
                    self._state.values.pop(key, None)
                    self._state.expiries.pop(key, None)
                    return existed
                return 0
        raise NotImplementedError("Unsupported in-memory Redis Lua script")

    def publish(self, _channel: str, _message) -> int:
        return 0

    def close(self) -> None:
        return None


class _MemoryPubSub:
    async def subscribe(self, *_channels) -> None:
        return None

    async def unsubscribe(self, *_channels) -> None:
        return None

    async def get_message(self, *args, **kwargs):
        return None

    async def aclose(self) -> None:
        return None


class _MemoryAsyncRedis:
    def __init__(self, sync_client: _MemorySyncRedis) -> None:
        self._sync_client = sync_client

    def pubsub(self) -> _MemoryPubSub:
        return _MemoryPubSub()

    async def aclose(self) -> None:
        return None


def _use_memory_redis() -> bool:
    return settings.REDIS_URL == "memory://"


def redis_key(*parts: object) -> str:
    prefix = settings.REDIS_KEY_PREFIX.strip(":")
    suffix = ":".join(str(part).strip(":") for part in parts if str(part))
    return f"{prefix}:{suffix}" if suffix else prefix


def get_sync_redis_client():
    global _sync_client
    if _sync_client is None:
        if _use_memory_redis():
            _sync_client = _MemorySyncRedis(_memory_state)
        else:
            _sync_client = sync_redis.from_url(settings.REDIS_URL)
    return _sync_client


def get_async_redis_client():
    global _async_client
    if _async_client is None:
        if _use_memory_redis():
            _async_client = _MemoryAsyncRedis(get_sync_redis_client())
        else:
            _async_client = async_redis.from_url(settings.REDIS_URL)
    return _async_client


def close_sync_redis_client() -> None:
    global _sync_client
    if _sync_client is None:
        return
    _sync_client.close()
    _sync_client = None


async def close_async_redis_client() -> None:
    global _async_client
    if _async_client is None:
        return
    await _async_client.aclose()
    _async_client = None


atexit.register(close_sync_redis_client)
