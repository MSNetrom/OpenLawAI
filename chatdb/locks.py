"""Distributed conversation locking for chat operations."""
from __future__ import annotations

import asyncio
import logging
import uuid

from config.redis_client import get_sync_redis_client, redis_key

logger = logging.getLogger(__name__)


class ConversationProcessingLock:
    """
    Prevents duplicate processing of the same conversation.
    
    Uses Redis SET NX with TTL and an owner token to ensure only one chat
    request processes a conversation at a time. Owner-checked Lua scripts
    guarantee that only the holder can refresh or release the lock.
    
    Usage:
        lock = ConversationProcessingLock(conversation_id)
        if not await lock.try_acquire():
            return error_response("Conversation already being processed")
        try:
            # ... chat processing ...
            await lock.refresh()   # periodically renew TTL
        finally:
            await lock.release()
    """

    # Lua: compare-and-expire (owner-checked TTL renewal)
    _LUA_REFRESH = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('expire', KEYS[1], ARGV[2])
else
  return 0
end
"""

    # Lua: compare-and-delete (owner-checked release)
    _LUA_RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
else
  return 0
end
"""

    def __init__(self, conversation_id, ttl: int = 1800):
        """
        Args:
            conversation_id: UUID of the conversation to lock
            ttl: Lock auto-expires after this many seconds (default: 30 min)
        """
        self.redis_client = get_sync_redis_client()
        self.lock_key = redis_key("chat_processing", conversation_id)
        self.ttl = ttl
        self.owner_token = uuid.uuid4().hex
        self._acquired = False

    async def _close_client(self) -> None:
        return None

    async def try_acquire(self) -> bool:
        """
        Try to acquire the lock. Returns True if acquired, False if already locked.
        Non-blocking - returns immediately.
        """
        acquired = await asyncio.to_thread(
            self.redis_client.set, self.lock_key, self.owner_token, nx=True, ex=self.ttl
        )
        if acquired:
            self._acquired = True
            logger.info("Acquired conversation lock key=%s ttl=%d", self.lock_key, self.ttl)
            return True
        await self._close_client()
        logger.info("Conversation already locked key=%s", self.lock_key)
        return False

    async def refresh(self) -> bool:
        """Renew the lock TTL. Returns True if refreshed, False if lock is no longer ours."""
        if not self._acquired:
            return False
        result = await asyncio.to_thread(
            self.redis_client.eval, self._LUA_REFRESH, 1, self.lock_key, self.owner_token, str(self.ttl)
        )
        if result:
            return True
        logger.warning("Lock refresh failed (not owner) key=%s", self.lock_key)
        return False

    async def release(self) -> None:
        """Release the lock and close the Redis client."""
        try:
            if self._acquired:
                result = await asyncio.to_thread(
                    self.redis_client.eval, self._LUA_RELEASE, 1, self.lock_key, self.owner_token
                )
                if result:
                    logger.info("Released conversation lock key=%s", self.lock_key)
                else:
                    logger.warning("Lock release failed (not owner) key=%s", self.lock_key)
        finally:
            self._acquired = False
            await self._close_client()
