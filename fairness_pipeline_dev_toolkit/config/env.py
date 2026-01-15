"""Environment variable configuration."""

import os
from typing import Optional


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get boolean from environment variable.

    Args:
        key: Environment variable name
        default: Default value if variable is not set

    Returns:
        Boolean value parsed from environment variable

    Examples:
        >>> os.environ['MY_FLAG'] = 'true'
        >>> get_env_bool('MY_FLAG', default=False)
        True
        >>> get_env_bool('NOT_SET', default=False)
        False
    """
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def get_env_int(key: str, default: Optional[int] = None) -> Optional[int]:
    """
    Get integer from environment variable.

    Args:
        key: Environment variable name
        default: Default value if variable is not set or invalid

    Returns:
        Integer value parsed from environment variable, or default if not set/invalid

    Examples:
        >>> os.environ['MY_NUM'] = '42'
        >>> get_env_int('MY_NUM', default=30)
        42
        >>> get_env_int('NOT_SET', default=30)
        30
        >>> get_env_int('INVALID', default=30)
        30
    """
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


# Common environment variables
FAIRPIPE_CONFIG_PATH = "FAIRPIPE_CONFIG_PATH"
FAIRPIPE_MIN_GROUP_SIZE = "FAIRPIPE_MIN_GROUP_SIZE"
FAIRPIPE_MLFLOW_EXPERIMENT = "FAIRPIPE_MLFLOW_EXPERIMENT"
