from tiny_transformer import __version__


def test_package_exposes_semantic_version() -> None:
    assert __version__ == "1.0.0"
