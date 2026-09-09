"""Crash-safe, checksummed local storage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_checksum(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, canonical_json(value) + "\n")


class JSONLStorage(Generic[T]):
    """Atomic checkpoint per record, with a derived JSONL view."""

    def __init__(self, filepath: str | Path, schema: type[T]):
        self.filepath = Path(filepath)
        self.schema = schema
        run_root = (
            self.filepath.parent.parent
            if self.filepath.parent.name in {"responses", "judgments"}
            else self.filepath.parent
        )
        self.records_dir = run_root / "checkpoints" / self.filepath.stem
        self.records_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def identity(record: BaseModel) -> str:
        values = record.model_dump()
        keys = [str(values.get(k, "")) for k in ("run_id", "split", "pattern_id", "sample_index")]
        return "__".join(part.replace("/", "_").replace("\\", "_") for part in keys)

    def _wrap(self, record: T) -> dict[str, Any]:
        data = canonical_json(record)
        return {"checksum": compute_checksum(data), "data": json.loads(data)}

    def append_record(self, record: T) -> None:
        validated = self.schema.model_validate(record)
        path = self.records_dir / f"{self.identity(validated)}.json"
        if path.exists():
            existing = self._read_checkpoint(path)
            if existing and canonical_json(existing) == canonical_json(validated):
                return
            raise ValueError(f"conflicting record already exists: {path.name}")
        atomic_write_json(path, self._wrap(validated))
        self.materialize()

    def _read_wrapped(self, wrapped: dict[str, Any]) -> T | None:
        try:
            data = wrapped["data"]
            data_text = data if isinstance(data, str) else canonical_json(data)
            if compute_checksum(data_text) != wrapped["checksum"]:
                return None
            return self.schema.model_validate_json(data_text)
        except (KeyError, TypeError, ValueError):
            return None

    def _read_checkpoint(self, path: Path) -> T | None:
        try:
            return self._read_wrapped(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return None

    def load_valid_records(self) -> list[T]:
        records = [self._read_checkpoint(path) for path in sorted(self.records_dir.glob("*.json"))]
        valid = [record for record in records if record is not None]
        if not valid and self.filepath.exists():
            for line in self.filepath.read_text(encoding="utf-8").splitlines():
                try:
                    record = self._read_wrapped(json.loads(line))
                except json.JSONDecodeError:
                    record = None
                if record is not None:
                    valid.append(record)
        deduped: dict[str, T] = {}
        for record in valid:
            identity = self.identity(record)
            if identity in deduped and canonical_json(record) != canonical_json(deduped[identity]):
                raise ValueError(f"duplicate sample ID has conflicting contents: {identity}")
            deduped[identity] = record
        return [deduped[key] for key in sorted(deduped)]

    def materialize(self) -> None:
        lines = [canonical_json(self._wrap(record)) for record in self.load_valid_records()]
        atomic_write_text(self.filepath, "\n".join(lines) + ("\n" if lines else ""))

    def valid_identities(self) -> set[str]:
        return {self.identity(record) for record in self.load_valid_records()}
