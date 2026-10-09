import zipfile

from rag_app.transfer import build_bundle


def test_transfer_bundle_excludes_secrets_and_includes_checksums(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "source.py").write_text("print('ok')", encoding="utf-8")
    (root / ".env").write_text("SECRET=value", encoding="utf-8")
    (root / ".ruff_cache").mkdir()
    (root / ".ruff_cache" / "state").write_text("local", encoding="utf-8")
    output = tmp_path / "bundle.zip"
    archive = build_bundle(root, output)
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        assert "traceable-pdf-rag/source.py" in names
        assert "traceable-pdf-rag/SHA256SUMS" in names
        assert "traceable-pdf-rag/.env" not in names
        assert "traceable-pdf-rag/.ruff_cache/state" not in names
