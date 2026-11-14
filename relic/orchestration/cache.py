"""Node-level caching utilities for orchestration execution."""

from __future__ import annotations

import hashlib
import hmac
import os
import pickle
import re
import struct
import tempfile
import warnings
from pathlib import Path
from typing import Any, Optional


class NodeCache:
    """Cache trusted-local node outputs behind a fail-closed envelope.

    The envelope detects accidental corruption and binds an entry to its cache
    key before pickle deserialization. It is not an authenticity boundary for
    adversarial or shared cache files because its digest is not keyed.
    """

    _KEY_PATTERN = re.compile(r"[0-9a-f]{8}_[0-9a-f]{16}\Z")
    _MAGIC = b"RELICNC1"
    _SCHEMA_VERSION = 1
    _HEADER = struct.Struct("!8sB25sQ32s")

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def hash_object(obj: Any) -> str:
        """Return a stable hash for a Python object payload."""
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _node_token(node_id: str) -> str:
        return hashlib.sha256(node_id.encode("utf-8")).hexdigest()[:8]

    def compute_key(
        self,
        node_id: str,
        inputs_hash: str,
        params_hash: str,
        code_hash: str,
    ) -> str:
        node_token = self._node_token(node_id)
        combined = f"{node_token}:{inputs_hash}:{params_hash}:{code_hash}"
        key_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
        return f"{node_token}_{key_hash}"

    def _path_for(self, key: str) -> Path:
        if not isinstance(key, str) or self._KEY_PATTERN.fullmatch(key) is None:
            raise ValueError(
                "NodeCache cache key must match "
                "'[0-9a-f]{8}_[0-9a-f]{16}'."
            )
        return self.cache_dir / f"{key}.pkl"

    @classmethod
    def _encode_envelope(cls, key: str, payload: bytes) -> bytes:
        key_bytes = key.encode("ascii")
        digest = hashlib.sha256(payload).digest()
        header = cls._HEADER.pack(
            cls._MAGIC,
            cls._SCHEMA_VERSION,
            key_bytes,
            len(payload),
            digest,
        )
        return header + payload

    @classmethod
    def _decode_envelope(cls, key: str, envelope: bytes) -> bytes:
        if len(envelope) < cls._HEADER.size:
            raise ValueError("cache envelope is truncated")
        magic, version, stored_key, payload_length, expected_digest = (
            cls._HEADER.unpack_from(envelope)
        )
        if magic != cls._MAGIC:
            raise ValueError("cache envelope magic is invalid")
        if version != cls._SCHEMA_VERSION:
            raise ValueError("cache envelope schema version is unsupported")
        if stored_key.rstrip(b"\x00") != key.encode("ascii"):
            raise ValueError("cache envelope key does not match the request")
        expected_length = cls._HEADER.size + payload_length
        if len(envelope) != expected_length:
            raise ValueError("cache envelope payload length is invalid")
        payload = envelope[cls._HEADER.size:]
        actual_digest = hashlib.sha256(payload).digest()
        if not hmac.compare_digest(actual_digest, expected_digest):
            raise ValueError("cache envelope payload digest is invalid")
        return payload

    @staticmethod
    def _discard_invalid_entry(path: Path, key: str, error: Exception) -> None:
        warnings.warn(
            f"NodeCache invalid cache entry for {key}: {error}",
            RuntimeWarning,
            stacklevel=3,
        )
        try:
            path.unlink(missing_ok=True)
        except OSError as unlink_error:
            warnings.warn(
                f"NodeCache could not remove invalid entry for {key}: "
                f"{unlink_error}",
                RuntimeWarning,
                stacklevel=3,
            )

    def _read_validated_payload(self, key: str, path: Path) -> Optional[bytes]:
        # An unreadable file (OSError) or an envelope that fails validation
        # (ValueError from _decode_envelope) is an invalid entry: it is
        # evicted with a RuntimeWarning and the lookup is a miss, so the node
        # is recomputed and no cached value is substituted.
        try:
            return self._decode_envelope(key, path.read_bytes())
        except (OSError, ValueError) as error:
            self._discard_invalid_entry(path, key, error)
            return None

    def get(self, key: str) -> Optional[Any]:
        path = self._path_for(key)
        if not path.exists():
            return None
        payload = self._read_validated_payload(key, path)
        if payload is None:
            return None
        # A digest-valid payload whose classes no longer resolve (module or
        # attribute removed since the entry was written) is a stale entry:
        # evicted, and the lookup is a miss. Any other error raised while
        # restoring the value is a defect of the cached type and propagates.
        try:
            return pickle.loads(payload)
        except (pickle.UnpicklingError, ImportError, AttributeError) as error:
            self._discard_invalid_entry(path, key, error)
            return None

    def put(self, key: str, value: Any) -> None:
        path = self._path_for(key)
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        envelope = self._encode_envelope(key, payload)
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{key}.",
                suffix=".tmp",
                dir=self.cache_dir,
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(envelope)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

    def has(self, key: str) -> bool:
        path = self._path_for(key)
        if not path.exists():
            return False
        return self._read_validated_payload(key, path) is not None

    def invalidate(self, node_id: str) -> int:
        """
        Invalidate all cache entries for a specific node.

        Parameters
        ----------
        node_id : str
            Node identifier to invalidate.

        Returns
        -------
        int
            Number of entries invalidated.
        """
        node_token = self._node_token(node_id)
        count = 0
        for path in self.cache_dir.glob(f"{node_token}_*.pkl"):
            path.unlink()
            count += 1
        return count

    def clear_all(self) -> int:
        """
        Clear all cache entries.

        Returns
        -------
        int
            Number of entries cleared.
        """
        count = 0
        for path in self.cache_dir.glob("*.pkl"):
            path.unlink()
            count += 1
        return count
