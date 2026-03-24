"""
Redis cache layer for real-time data and intermediate results.
"""

import json
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from config.settings import Settings
from utils.logger import get_logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False

logger = get_logger(__name__)


class CacheManager:
    """Redis-backed cache for real-time and hot data."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self._client = None
        self.available = False
        if not REDIS_AVAILABLE:
            logger.warning("redis package not installed — cache disabled")
            return
        try:
            self._client = redis.from_url(
                self.settings.redis.url,
                decode_responses=False,
                socket_connect_timeout=5,
            )
            self._client.ping()
            self.available = True
            logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}. Cache disabled.")
            self._client = None
            self.available = False

    def _key(self, namespace: str, identifier: str) -> str:
        return f"trading:{namespace}:{identifier}"

    # ------------------------------------------------------------------ #
    #  Generic get/set                                                     #
    # ------------------------------------------------------------------ #

    def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if not self.available:
            return False
        ttl = ttl or self.settings.redis.ttl_seconds
        try:
            serialized = json.dumps(value, default=str).encode()
            self._client.setex(self._key(namespace, key), ttl, serialized)
            return True
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")
            return False

    def get(self, namespace: str, key: str) -> Optional[Any]:
        if not self.available:
            return None
        try:
            raw = self._client.get(self._key(namespace, key))
            if raw is None:
                return None
            return json.loads(raw.decode())
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
            return None

    def delete(self, namespace: str, key: str) -> None:
        if not self.available:
            return
        try:
            self._client.delete(self._key(namespace, key))
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  DataFrame-specific helpers                                          #
    # ------------------------------------------------------------------ #

    def cache_dataframe(self, namespace: str, key: str, df: pd.DataFrame, ttl: Optional[int] = None) -> bool:
        """Cache a DataFrame as JSON."""
        if not self.available or df.empty:
            return False
        try:
            ttl = ttl or self.settings.redis.ttl_seconds
            payload = df.to_json(orient="split", date_format="iso")
            self._client.setex(self._key(namespace, key), ttl, payload.encode())
            return True
        except Exception as e:
            logger.debug(f"DataFrame cache failed: {e}")
            return False

    def get_dataframe(self, namespace: str, key: str) -> Optional[pd.DataFrame]:
        if not self.available:
            return None
        try:
            raw = self._client.get(self._key(namespace, key))
            if raw is None:
                return None
            df = pd.read_json(raw.decode(), orient="split")
            return df
        except Exception as e:
            logger.debug(f"DataFrame cache read failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Signal caching                                                      #
    # ------------------------------------------------------------------ #

    def cache_signal(self, symbol: str, signal: dict, ttl: int = 3600) -> None:
        self.set("signals", symbol, signal, ttl=ttl)

    def get_signal(self, symbol: str) -> Optional[dict]:
        return self.get("signals", symbol)

    def get_all_cached_signals(self) -> list[dict]:
        if not self.available:
            return []
        try:
            pattern = self._key("signals", "*")
            keys = self._client.keys(pattern)
            results = []
            for k in keys:
                raw = self._client.get(k)
                if raw:
                    results.append(json.loads(raw.decode()))
            return results
        except Exception as e:
            logger.debug(f"Get all signals failed: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Rate-limit state                                                    #
    # ------------------------------------------------------------------ #

    def increment_rate_counter(self, api_name: str, window_secs: int = 60) -> int:
        """Track API call counts for rate limiting. Returns current count."""
        if not self.available:
            return 0
        key = self._key("rate_limit", f"{api_name}:{window_secs}")
        try:
            pipe = self._client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window_secs)
            results = pipe.execute()
            return results[0]
        except Exception:
            return 0
