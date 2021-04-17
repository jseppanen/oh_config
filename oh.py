
import json
from functools import partial, wraps
from pathlib import Path

from typing import Any, Callable, Dict, Optional, List, TextIO, Union
from configparser import ConfigParser
from contextlib import contextmanager


class Config(dict):
    """Tree of configuration values.
    """

    def __getattr__(self, name: str) -> Any:
        """Convenience attribute access to config values."""
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"Configuration value not found: {name}")

    def load_str(self, txt: str) -> None:
        """Load configuration from string.

        Multiple calls update config incrementally.
        """
        config = ConfigParser()
        config.optionxform = str
        config.read_string(txt)
        self.load_config(config)

    def load_file(self, fd: Union[str, Path, TextIO]) -> None:
        """Load configuration from file.

        Multiple calls update config incrementally.
        """
        if isinstance(fd, (str, Path)):
            fd = open(fd)
        self.load_str(fd.read())

    def load_config(self, config: ConfigParser) -> None:
        if config.defaults():
            raise ParseError("Found config values outside of any section")
        get_depth = lambda item: len(item[0].split("."))
        for section, values in sorted(config.items(), key=get_depth):
            if section == "DEFAULT":
                continue
            parts = section.split(".")
            node = self
            for part in parts[:-1]:
                if part not in node:
                    raise ParseError("Error parsing config section {part}")
                else:
                    node = node[part]
            if not isinstance(node, dict):
                raise ParseError(f"Found conflicting values for {part}")
            # Set the default section
            node = node.setdefault(parts[-1], Config())
            for key, value in values.items():
                config_v = config.get(section, key)
                node[key] = try_load_json(config_v)


class ParseError(ValueError):
    pass


class ConfigView(Dict[str, Any]):
    """View of a Config (sub)section.
    Used for traversing subsections with a context manager.
    """
    def __init__(self, config: Config):
        self._config = config
        self._view_path: List[str] = []

    def __getattr__(self, name: str) -> Any:
        """Convenience attribute access to config values."""
        return getattr(self._node, name)

    def __getitem__(self, name: str) -> Any:
        return self._node[name]

    def __iter__(self):
        return iter(self._node)

    def __len__(self):
        return len(self._node)

    def __contains__(self, name: str) -> bool:
        return name in self._node

    def __repr__(self) -> str:
        if not self._view_path:
            return f"ConfigView: {self._config}"
        else:
            return f"ConfigView[{'.'.join(self._view_path)}]: {self._node}"

    @property
    def _node(self) -> Config:
        node = self._config
        for key in self._view_path:
            node = node[key]
        return node

    @contextmanager
    def enter(self, section: str):
        """Traverse config tree down to view to a subsection."""
        parts = section.split(".")
        for part in parts:
            if part not in self._node:
                raise KeyError(f"No such configuration section: {part}")
            elif not isinstance(self._node[part], Config):
                raise KeyError(f"Not a section: {part}: {self._node[part]}")
            else:
                self._view_path.append(part)
        try:
            yield
        finally:
            for part in parts[::-1]:
                popped = self._view_path.pop()
                assert popped == part, "push/pop invariant failed"


def register(
    func_or_class_or_name: Optional[Union[Callable, str]] = None,
    name: Optional[str] = None,
) -> Callable:
    """Register functions to use configurable arguments when called.

    Decorating a class is convenience for decorating its __init__ method.
    """
    if isinstance(func_or_class_or_name, str):
        # @oh.register("foobar")
        return partial(register, name=func_or_class_or_name)

    elif isinstance(func_or_class_or_name, type):
        # decorating a class is convenience for decorating its __init__ method
        cls = func_or_class_or_name
        if not hasattr(cls, "__init__"):
            raise TypeError(f"cannot register classes without __init__: {cls}")
        wrapper = register(cls.__init__, name=name or cls.__name__)
        setattr(cls, "__init__", wrapper)
        return cls

    if func_or_class_or_name is None:
        # @oh.register
        return partial(register, name=name)

    # @oh.register()
    func = func_or_class_or_name
    if not name:
        if func.__name__ != "__init__":
            name = func.__name__
        else:
            # Use parent class's name as default name for __init__ functions
            assert func.__qualname__.endswith(".__init__"), "funny qualname"
            name = func.__qualname__.split(".")[-2]

    @wraps(func)
    def wrapper(*args, **overrides):
        # fetch parameters from configuration and apply overrides
        global config
        if name in config:
            kwargs = dict(config[name], **overrides)
        else:
            kwargs = overrides
        return func(*args, **kwargs)
    return wrapper


def try_load_json(value: str) -> Any:
    """Load a JSON string if possible, otherwise default to original value."""
    try:
        return json.loads(value)
    except Exception:
        return value


# default global singleton configuration
config = ConfigView(Config())
