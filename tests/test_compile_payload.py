from __future__ import annotations

import quack._compile_payload as payload
from quack._compile_payload import (
    epilogue_source_digest,
    load_epilogue_from_source,
    make_epilogue_cache_key,
    make_epilogue_source_marker,
    set_epilogue_source_cache_key,
)


def test_epilogue_source_marker_separates_symbol_name_from_epilogue_key(tmp_path, monkeypatch):
    def actual_epilogue(*args, **kwargs):
        return "parent"

    source = "def actual_epilogue(*args, **kwargs):\n    return 'ok'\n"
    marker = make_epilogue_source_marker("actual_epilogue", "logical-cache-key", source)
    parent_cache_key = set_epilogue_source_cache_key(actual_epilogue, source)

    assert marker["symbol_name"] == "actual_epilogue"
    assert marker["epilogue_key"] == "logical-cache-key"
    assert marker["cache_key"] == make_epilogue_cache_key("actual_epilogue", source)
    assert marker["cache_key"] == parent_cache_key
    assert epilogue_source_digest(source) in marker["cache_key"]

    monkeypatch.setattr(payload.tempfile, "gettempdir", lambda: str(tmp_path))
    epilogue_fn = load_epilogue_from_source(marker)

    assert epilogue_fn() == "ok"
    assert epilogue_fn.__quack_cache_key__ == parent_cache_key


def test_epilogue_cache_key_changes_with_source_digest():
    source_a = "def actual_epilogue(*args, **kwargs):\n    return 'a'\n"
    source_b = "def actual_epilogue(*args, **kwargs):\n    return 'b'\n"

    assert make_epilogue_cache_key("actual_epilogue", source_a) != make_epilogue_cache_key(
        "actual_epilogue", source_b
    )


def test_load_epilogue_from_source_replaces_stale_partial_module(tmp_path, monkeypatch):
    source = "def actual_epilogue(*args, **kwargs):\n    return 'complete'\n"
    marker = make_epilogue_source_marker("actual_epilogue", "logical-cache-key", source)
    module_dir = tmp_path / "quack_generated_epilogues"
    module_dir.mkdir()
    module_path = module_dir / f"quack_generated_epilogue_{marker['source_digest'][:16]}.py"
    module_path.write_text("def actual_epilogue(")

    monkeypatch.setattr(payload.tempfile, "gettempdir", lambda: str(tmp_path))
    epilogue_fn = load_epilogue_from_source(marker)

    assert epilogue_fn() == "complete"
    assert module_path.read_text() == source
    assert not list(module_dir.glob("*.tmp"))
