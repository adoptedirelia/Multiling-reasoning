__all__ = [
    "ErrorSimConfig",
    "ModelEngineConfig",
    "load_config",
]


def __getattr__(name):
    if name in __all__:
        from .error_prop.config import ErrorSimConfig, ModelEngineConfig, load_config

        values = {
            "ErrorSimConfig": ErrorSimConfig,
            "ModelEngineConfig": ModelEngineConfig,
            "load_config": load_config,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
