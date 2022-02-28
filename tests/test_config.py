from configuration.config import Config

config1 = """
[paths]
path1 = "/tmp/path1"
path2 = "/tmp/path2"

[test]
path = ${paths.path2}
"""

config2 = """
[paths]
path2 = "/tmp/path3"
"""


def test_config(tmp_path):

    path1 = tmp_path / "config1.cfg"
    path2 = tmp_path / "config2.cfg"

    path1.write_text(config1)
    path2.write_text(config2)

    config = Config().multi_from_disk(path1, interpolate=False)

    assert config["test"]["path"] == "${paths.path2}"

    config = Config().multi_from_disk(path1)

    assert config["test"]["path"] == "/tmp/path2"

    config = Config().multi_from_disk(path1, path2)

    assert config["test"]["path"] == "/tmp/path3"
