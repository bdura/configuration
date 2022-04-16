from configuration import config

config1 = """
[paths]
path1 = "/tmp/path1"
path2 = "/tmp/path2"

[test]
path = "${paths.path2}"
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

    cfg = config.Config.multi_from_disk(path1, interpolate=False)

    assert cfg["test"]["path"] == "${paths.path2}"

    # cfg = config.Config.multi_from_disk(path1)

    # assert cfg["test"]["path"] == "/tmp/path2"

    # cfg = config.Config.multi_from_disk(path1, path2)

    # assert cfg["test"]["path"] == "/tmp/path3"


def test_check_last():
    sequence = [i for i in range(10)]
    res = [last for _, last in config.check_last(sequence)]

    assert not any(res[:-1])
    assert res[-1]


def test_merge():
    d1 = {
        "a": 1,
        "b": {"a": 2, "b": 2},
    }
    d2 = {
        "a": {"a": 1, "b": 2},
        "b": {"a": 1},
        "c": 1,
    }

    m = {
        "a": {"a": 1, "b": 2},
        "b": {"a": 1, "b": 2},
        "c": 1,
    }

    merged = config.merge(d1, d2)

    assert merged == m


def test_config_merge():

    d1 = {
        "a": 1,
        "b": {"a": 2, "b": 2},
    }
    d2 = {
        "a": {"a": 1, "b": 2},
        "b": {"a": 1},
        "c": 1,
    }

    m = {
        "a": {"a": 1, "b": 2},
        "b": {"a": 1, "b": 2},
        "c": 1,
    }

    c1 = config.Config(d1)

    assert c1.merge(d2).data == m
