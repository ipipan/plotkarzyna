from dataclasses import dataclass, field
from pathlib import Path

import pkg_resources
import yaml

from dotenv import dotenv_values

try:
    local_config = dotenv_values(".env")
except FileNotFoundError:
    local_config = dict()


def get_config(config_path: Path = None) -> 'Config':
    if config_path is None:
        config_path = pkg_resources.resource_filename(
            'python_template',
            'config.yaml'
        )
    elif config_path.exists():
        pass
    else:
        raise FileNotFoundError(f'Config file {config_path} not found')

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)


@dataclass
class YamlConfig:
    is_working: bool = False


@dataclass
class Config:
    yaml_config: YamlConfig = field(default_factory=YamlConfig)

    def __post_init__(self):
        self.yaml_config = YamlConfig(**self.yaml_config)
