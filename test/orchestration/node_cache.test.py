"""NodeCache unit tests."""

from __future__ import annotations

import operator
import pickle
from pathlib import Path

import pytest

import relic.orchestration.cache as cache_module
from relic.orchestration.cache import NodeCache


def _cache_key(cache: NodeCache, node_id: str = "node_1") -> str:
    return cache.compute_key(
        node_id,
        cache.hash_object({"x": 1}),
        cache.hash_object({"p": 2}),
        "code_v1",
    )


def _write_execution_marker(marker_path: str) -> None:
    Path(marker_path).write_text("executed", encoding="utf-8")


class _MaliciousPickle:
    def __init__(self, marker_path: Path) -> None:
        self.marker_path = marker_path

    def __reduce__(self):
        return (_write_execution_marker, (str(self.marker_path),))


def test_node_cache_roundtrip(tmp_path) -> None:
    cache = NodeCache(tmp_path)
    inputs_hash = cache.hash_object({"x": 1})
    params_hash = cache.hash_object({"p": 2})
    code_hash = "code_v1"

    key = cache.compute_key("node_1", inputs_hash, params_hash, code_hash)
    assert cache.has(key) is False

    payload = {"result": 123}
    cache.put(key, payload)
    assert cache.has(key) is True
    assert cache.get(key) == payload


def test_node_cache_invalidate_and_clear(tmp_path) -> None:
    cache = NodeCache(tmp_path)
    inputs_hash = cache.hash_object({"x": 1})
    params_hash = cache.hash_object({"p": 2})
    code_hash = "code_v1"

    key_a = cache.compute_key("node_a", inputs_hash, params_hash, code_hash)
    key_b = cache.compute_key("node_b", inputs_hash, params_hash, code_hash)

    cache.put(key_a, {"a": 1})
    cache.put(key_b, {"b": 2})

    assert cache.invalidate("node_a") == 1
    assert cache.has(key_a) is False
    assert cache.has(key_b) is True

    assert cache.clear_all() == 1
    assert cache.has(key_b) is False


@pytest.mark.parametrize(
    "invalid_key",
    [
        "../escape",
        "..\\escape",
        "C:\\escape",
        "not-a-cache-key",
        "deadbeef_0123456789abcdef.pkl",
    ],
)
def test_node_cache_rejects_invalid_keys(tmp_path, invalid_key) -> None:
    cache = NodeCache(tmp_path)

    with pytest.raises(ValueError, match="cache key"):
        cache.put(invalid_key, {"result": 1})
    with pytest.raises(ValueError, match="cache key"):
        cache.get(invalid_key)
    with pytest.raises(ValueError, match="cache key"):
        cache.has(invalid_key)


def test_node_cache_rejects_raw_pickle_without_deserializing(tmp_path) -> None:
    cache = NodeCache(tmp_path / "cache")
    key = _cache_key(cache)
    marker = tmp_path / "pickle_executed.txt"
    cache._path_for(key).write_bytes(
        pickle.dumps(_MaliciousPickle(marker), protocol=pickle.HIGHEST_PROTOCOL)
    )

    with pytest.warns(RuntimeWarning, match="invalid cache entry"):
        assert cache.get(key) is None

    assert marker.exists() is False
    assert cache._path_for(key).exists() is False


def test_node_cache_rejects_truncated_envelope(tmp_path) -> None:
    cache = NodeCache(tmp_path)
    key = _cache_key(cache)
    cache.put(key, {"result": 123})
    path = cache._path_for(key)
    payload = path.read_bytes()
    path.write_bytes(payload[:-1])

    with pytest.warns(RuntimeWarning, match="invalid cache entry"):
        assert cache.get(key) is None

    assert path.exists() is False


def test_node_cache_rejects_payload_digest_mismatch(tmp_path) -> None:
    cache = NodeCache(tmp_path)
    key = _cache_key(cache)
    cache.put(key, {"result": 123})
    path = cache._path_for(key)
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0x01
    path.write_bytes(payload)

    with pytest.warns(RuntimeWarning, match="invalid cache entry"):
        assert cache.get(key) is None

    assert path.exists() is False


def test_node_cache_rejects_envelope_copied_to_another_key(tmp_path) -> None:
    cache = NodeCache(tmp_path)
    key_a = _cache_key(cache, "node_a")
    key_b = _cache_key(cache, "node_b")
    cache.put(key_a, {"result": 123})
    cache._path_for(key_b).write_bytes(cache._path_for(key_a).read_bytes())

    with pytest.warns(RuntimeWarning, match="invalid cache entry"):
        assert cache.get(key_b) is None

    assert cache._path_for(key_b).exists() is False
    assert cache.get(key_a) == {"result": 123}


class _RestoreDefect:
    """Cached value whose restore raises: a code defect, not a stale entry."""

    def __reduce__(self):
        return (operator.truediv, (1, 0))


def test_node_cache_evicts_entry_whose_classes_no_longer_resolve(tmp_path) -> None:
    cache = NodeCache(tmp_path)
    key = _cache_key(cache)
    # A digest-valid entry written by an environment whose module is gone.
    stale_payload = b"crelic_test_removed_cache_module\nRemovedResult\n."
    cache._path_for(key).write_bytes(NodeCache._encode_envelope(key, stale_payload))

    with pytest.warns(RuntimeWarning, match="invalid cache entry"):
        assert cache.get(key) is None

    assert cache._path_for(key).exists() is False


def test_node_cache_propagates_restore_defect_of_a_valid_entry(tmp_path) -> None:
    cache = NodeCache(tmp_path)
    key = _cache_key(cache)
    cache.put(key, _RestoreDefect())

    with pytest.raises(ZeroDivisionError):
        cache.get(key)

    assert cache._path_for(key).exists() is True


def test_node_cache_propagates_envelope_decoder_defect(tmp_path, monkeypatch) -> None:
    cache = NodeCache(tmp_path)
    key = _cache_key(cache)
    cache.put(key, {"result": 123})

    def _decoder_defect(cls, key, envelope):
        raise TypeError("envelope decoder defect")

    monkeypatch.setattr(NodeCache, "_decode_envelope", classmethod(_decoder_defect))

    with pytest.raises(TypeError, match="envelope decoder defect"):
        cache.get(key)

    assert cache._path_for(key).exists() is True


def test_node_cache_replace_failure_preserves_previous_entry(
    tmp_path,
    monkeypatch,
) -> None:
    cache = NodeCache(tmp_path)
    key = _cache_key(cache)
    cache.put(key, {"result": "old"})

    def _replace_failure(source, destination) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(cache_module.os, "replace", _replace_failure)
    with pytest.raises(OSError, match="replace failed"):
        cache.put(key, {"result": "new"})

    assert cache.get(key) == {"result": "old"}
    assert list(tmp_path.glob("*.tmp")) == []
