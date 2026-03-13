from pathlib import Path

from inline_snapshot import snapshot

from project.data import load_sql_files


def test_load_sql_files_selectively_loads_sql_files(tmp_path: Path):
    """The SQL file reader should only consider SQL files."""
    (tmp_path / "a.sql").write_text("SELECT 1;", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Text", encoding="utf-8")
    (tmp_path / "c.sql").write_text("SELECT 2;", encoding="utf-8")
    inputs = str(tmp_path / "*.sql")
    expected = snapshot({"a": "SELECT 1;", "c": "SELECT 2;"})

    actual = load_sql_files(inputs)

    assert dict(actual) == expected


def test_load_sql_files_returns_empty_dict_for_no_files(tmp_path: Path):
    """The loader should not fail if there are no files."""
    expected = snapshot({})
    actual = load_sql_files(str(tmp_path / "*.sql"))
    assert dict(actual) == expected


def test_load_sql_files_finds_files_recursively(tmp_path: Path):
    """This allows more elaborate file storing in groups of sub-directories."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "test.sql").write_text("SELECT 42;", encoding="utf-8")
    expected = snapshot({"test": "SELECT 42;"})

    actual = load_sql_files(str(tmp_path / "**/*.sql"))

    assert dict(actual) == expected


def test_load_sql_files_has_deterministic_order(tmp_path: Path):
    """Order is essential to prevent differences in joining data downstream."""
    (tmp_path / "b.sql").write_text("B", encoding="utf-8")
    (tmp_path / "a.sql").write_text("A", encoding="utf-8")
    expected = snapshot(["a", "b"])

    actual = load_sql_files(str(tmp_path / "*.sql"))

    assert list(actual.keys()) == expected
