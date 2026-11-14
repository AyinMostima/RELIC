"""Runtime configuration loading helpers shared by CLI and API."""

import logging
from pathlib import Path
from typing import Any, cast

import relic.core.config as core_config
from relic.core.config.loader import CONFIG_PATH_ENV_VAR

logger = logging.getLogger(__name__)


def load_runtime_config(
    config_path: str | None = None,
    env_var: str = CONFIG_PATH_ENV_VAR,
) -> tuple[dict[str, Any], Any | None, Path | None]:
    """
    Load and validate runtime configuration using core config utilities.

    Parameters
    ----------
    config_path : Optional[str]
        Explicit configuration path provided by caller.
    env_var : str
        Environment variable name for default config path.

    Returns
    -------
    Tuple[Dict[str, Any], Optional[Any], Optional[Path]]
        (config, validation_result, config_file)

    Source resolution is the one RELIC rule, ``core.config.resolve_config_source``:
    an explicit ``config_path``, else the file named by ``env_var``; a named
    source that does not exist raises FileNotFoundError. Without a named source
    the first existing ``DEFAULT_CONFIG_PATHS`` file is loaded and reported as
    ``config_file``; when none exists the global default configuration
    (``RELICConfig()`` defaults) is used, ``config_file`` is None and the
    choice is logged. Every branch loads its source directly, so the reported
    source is always the one that was read (the cached ``get_global_config``
    instance is not consulted).
    """
    loaded_config: Any
    config_file: Path | None = core_config.resolve_config_source(config_path, env_var=env_var)
    if config_file is not None:
        loaded_config = core_config.load_config(
            str(config_file),
            validate=False,
        )
        logger.info("Loaded configuration from: %s", config_file)
    else:
        loaded_config = core_config.RELICConfig()
        logger.info(
            "Using the global default configuration (RELICConfig defaults): no "
            "configuration file was named and no default location exists."
        )

    validation_result = core_config.validate_configuration(loaded_config)

    if isinstance(loaded_config, dict):
        config = loaded_config
    elif hasattr(loaded_config, "model_dump"):
        config = cast(dict[str, Any], loaded_config.model_dump())
    else:
        raise TypeError(
            "Runtime configuration must be a mapping or a RELICConfig model, "
            f"got {type(loaded_config).__name__}."
        )

    return config, validation_result, config_file
