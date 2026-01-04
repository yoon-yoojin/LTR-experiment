"""Configuration management utilities."""
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration class for managing experiment settings."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Dictionary containing configuration
        """
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            *keys: Configuration keys (e.g., 'model', 'pairwise', 'name')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def update(self, *keys: str, value: Any) -> None:
        """Update configuration value.

        Args:
            *keys: Configuration keys
            value: New value
        """
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

    def save(self, save_path: str = None) -> None:
        """Save configuration to YAML file.

        Args:
            save_path: Path to save configuration (default: original path)
        """
        save_path = save_path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def __repr__(self) -> str:
        return f"Config({self.config})"
