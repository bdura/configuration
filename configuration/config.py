from pathlib import Path
from typing import Iterable, List, Sequence, Union, Any

from pydantic import validate_arguments

from copy import deepcopy

from .typing import DictStrAny

import tomli
import tomli_w


def is_leaf(v: Any) -> bool:
    return not isinstance(v, dict)


def check_last(sequence: Sequence) -> Iterable:
    last_i = len(sequence) - 1

    for i, item in enumerate(sequence):
        yield item, i == last_i


def walk(d: DictStrAny, keys: List[str] = []):
    for key, value in d.items():
        if is_leaf(value):
            yield tuple(keys + [key]), value
        else:
            yield from walk(value, keys=keys + [key])


def merge(d1: DictStrAny, d2: DictStrAny) -> DictStrAny:

    data = deepcopy(d1)

    for keys, value in walk(d2):
        data_ = data
        for key, last in check_last(keys):
            if last:
                data_[key] = value
            else:
                if key not in data_ or is_leaf(data_[key]):
                    data_[key] = dict()
                elif last:
                    data_[key] = value
                data_ = data_[key]

    return data


class Config:
    def __init__(self, data, *, interpolate: bool = True):
        if isinstance(data, Config):
            data = data.data
        self.data = data

    def items(self):
        return self.data.items()

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def merge(self, config: Union["Config", DictStrAny]) -> "Config":
        if isinstance(config, Config):
            config = config.data
        return Config(merge(self.data, config))

    def interpolate(self) -> "Config":
        pass

    def to_str(self) -> str:
        return tomli_w.dumps(self.data)

    @validate_arguments
    def to_disk(self, path: Path) -> None:
        toml = self.to_str()
        path.write_text(toml)

    @classmethod
    def from_str(cls, toml: str, interpolate: bool = True) -> DictStrAny:
        data = tomli.loads(toml)
        config = Config(data)
        return config

    @classmethod
    def from_disk(cls, path: Path, interpolate: bool = True) -> DictStrAny:
        config = cls.from_str(
            toml=path.read_text(),
            interpolate=interpolate,
        )
        return config

    @classmethod
    def merge_configs(
        cls,
        *configurations: List[Union["Config", DictStrAny]],
        interpolate: bool = True
    ) -> "Config":

        config: "Config" = Config(configurations[0], interpolate=False)

        for cfg in configurations[1:]:
            config = config.merge(cfg)

        if interpolate:
            config = config.interpolate()

        return config

    @classmethod
    def multi_from_disk(
        cls,
        *paths: List[Path],
        interpolate: bool = True,
        first_interpolation: bool = False
    ) -> "Config":
        configs = [
            Config.from_disk(path, interpolate=first_interpolation) for path in paths
        ]
        return cls.merge_configs(*configs, interpolate=interpolate)
