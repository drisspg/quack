from __future__ import annotations

import quack._compile_payload as payload
from quack._compile_payload import (
    epilogue_source_digest,
    load_epilogue_from_source,
    make_epilogue_cache_key,
    make_epilogue_source_marker,
    set_epilogue_source_cache_key,
)


def _epilogue_source(return_value: str) -> str:
    return f"def actual_epilogue(*args, **kwargs):\n    return {return_value!r}\n"


def test_epilogue_source_marker_loads_symbol_with_digest_cache_key(tmp_path, monkeypatch):
    def actual_epilogue(*args, **kwargs):
        return "parent"

    source = _epilogue_source("ok")
    marker = make_epilogue_source_marker("actual_epilogue", source)
    parent_cache_key = set_epilogue_source_cache_key(actual_epilogue, source)

    assert payload.is_epilogue_source_marker(marker)
    assert marker == {
        payload.EPILOGUE_SOURCE_TAG: True,
        "symbol_name": "actual_epilogue",
        "source": source,
    }
    assert parent_cache_key == make_epilogue_cache_key("actual_epilogue", source)
    assert epilogue_source_digest(source) in parent_cache_key

    monkeypatch.setattr(payload.tempfile, "gettempdir", lambda: str(tmp_path))
    epilogue_fn = load_epilogue_from_source(marker)

    assert epilogue_fn() == "ok"
    assert epilogue_fn.__quack_cache_key__ == parent_cache_key


def test_epilogue_cache_key_changes_with_symbol_and_source_digest():
    source_a = _epilogue_source("a")
    source_b = _epilogue_source("b")

    assert make_epilogue_cache_key("actual_epilogue", source_a) != make_epilogue_cache_key(
        "actual_epilogue", source_b
    )
    assert make_epilogue_cache_key("actual_epilogue", source_a) != make_epilogue_cache_key(
        "other_epilogue", source_a
    )


def test_load_epilogue_from_source_replaces_stale_partial_module(tmp_path, monkeypatch):
    source = _epilogue_source("complete")
    marker = make_epilogue_source_marker("actual_epilogue", source)
    module_dir = tmp_path / "quack_generated_epilogues"
    module_dir.mkdir()
    module_path = module_dir / f"quack_generated_epilogue_{epilogue_source_digest(source)[:16]}.py"
    module_path.write_text("def actual_epilogue(")
    assert module_path.read_text() == "def actual_epilogue("

    monkeypatch.setattr(payload.tempfile, "gettempdir", lambda: str(tmp_path))
    epilogue_fn = load_epilogue_from_source(marker)

    assert epilogue_fn() == "complete"
    assert epilogue_fn.__quack_cache_key__ == make_epilogue_cache_key(
        "actual_epilogue", source
    )
    assert module_path.read_text() == source
    assert not list(module_dir.glob("*.tmp"))
