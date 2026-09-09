from __future__ import annotations

import json

import pytest

from phase_a.schemas import GenerationRecord
from phase_a.storage import JSONLStorage


def test_atomic_storage_resume_and_corruption_recovery(tmp_path, generation_factory):
    storage = JSONLStorage(tmp_path / "responses" / "responses_smoke.jsonl", GenerationRecord)
    first = generation_factory()
    storage.append_record(first)
    assert storage.load_valid_records() == [first]
    storage.append_record(first)
    assert len(storage.load_valid_records()) == 1
    (storage.records_dir / "corrupt.json").write_text('{"checksum":"bad","data":', encoding="utf-8")
    assert storage.load_valid_records() == [first]
    with pytest.raises(ValueError, match="conflicting"):
        storage.append_record(generation_factory(response_text="different"))


def test_materialized_jsonl_corruption_does_not_destroy_checkpoints(tmp_path, generation_factory):
    storage = JSONLStorage(tmp_path / "responses" / "responses_smoke.jsonl", GenerationRecord)
    storage.append_record(generation_factory())
    storage.filepath.write_text("partial", encoding="utf-8")
    assert len(storage.load_valid_records()) == 1
    storage.materialize()
    assert json.loads(storage.filepath.read_text().splitlines()[0])["checksum"]
