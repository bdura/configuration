from typing import List
from thinc import config
from pathlib import Path


class Config(config.Config):
    @classmethod
    def merge_configs(
        cls, *configurations: List["Config"], interpolate: bool = True
    ) -> "Config":

        config: "Config" = configurations[0]

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
            Config().from_disk(path, interpolate=first_interpolation) for path in paths
        ]
        return cls.merge_configs(*configs, interpolate=interpolate)
