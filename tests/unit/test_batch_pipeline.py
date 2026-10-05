import json
from pathlib import Path

from grievance_anonymization.batch_pipeline import (
    BatchError,
    _hardcoded_policy,
    _line_bases,
    _load_config,
    _path,
    _resolve_config_file,
)


def test_path_resolution():
    root = Path("/tmp/test_root")
    assert _path("config.json", root) == Path("/tmp/test_root/config.json")
    assert _path("/abs/path.json", root) == Path("/abs/path.json")


def test_resolve_config_file(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"data_type": "complaints", "complaints": {}}))
    assert _resolve_config_file(cfg_file) == cfg_file
    assert _resolve_config_file(tmp_path) == cfg_file


def test_load_config(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps({"data_type": "complaints", "complaints": {"enabled": True}})
    )
    dataset = _load_config(cfg_file)
    assert dataset == {"enabled": True}


def test_line_bases():
    text = "Line 1\n  Line 2\nLine 3"
    bases = _line_bases(text)
    assert len(bases) == 3


def test_hardcoded_policy():
    salts = {}
    val, tech = _hardcoded_policy("Aadhaar", "123456789012", "col", salts)
    assert "9012" in val
    assert tech == "partial_mask"


def test_batch_error():
    err = BatchError("Test batch failure")
    assert str(err) == "Test batch failure"
