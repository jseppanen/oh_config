
import importlib
import json
from functools import partial
from pathlib import Path

from typing import Any, Callable, Dict, Optional, List, TextIO, Tuple, Union
from configparser import ConfigParser
from contextlib import contextmanager


class Config(dict):
    """Tree of configuration values.
    """

    def __call__(self, *pos_overrides, **kw_overrides) -> Any:
        """Call functions/types referenced in config.
        Only sections having special `@call` key can be called.
        """
        if "@call" not in self:
            raise TypeError("Not callable: no @call key")

        defaults = self.copy()
        func_name = defaults.pop("@call")
        func = resolve(func_name)
        args, kwargs = merge_args(defaults, *pos_overrides, **kw_overrides)
        return dispatch(func, args, kwargs)

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
        config = ConfigParser(
            delimiters=["="],
            comment_prefixes=["#"],
            inline_comment_prefixes=["#"],
            strict=True,
            empty_lines_in_values=False,
        )
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
                raise ParseError(f"Found conflicting values for {parts}")
            # Set the default section
            node = node.setdefault(parts[-1], Config())
            for key, value in values.items():
                # parse key
                if key.startswith("@"):
                    # special @ key
                    if key != "@call":
                        raise ParseError(f"Key is not supported: {repr(key)}")
                    else:
                        node[key] = value
                    continue
                elif isintegral(key):
                    # integral key, used for positional arguments
                    pos = int(key)
                    if pos < 0:
                        raise ParseError(f"Negative positions are not valid: {pos}")
                    key = str(pos)
                elif not str.isidentifier(key):
                    raise ParseError(f"Key is not valid: {repr(key)}")

                # parse value
                config_v = config.get(section, key)
                try:
                    parsed_value = json.loads(config_v)
                except Exception:
                    raise ParseError(f"Error parsing value of {key}: {config_v}")
                node[key] = parsed_value


class ParseError(ValueError):
    pass


def merge_args(defaults: Dict, *pos_overrides, **kw_overrides) -> Tuple[Tuple, Dict]:
    """Merge positional and keyword arguments."""
    # apply overrides
    merged = defaults.copy()
    merged.update({
        str(k): v for k, v in enumerate(pos_overrides)
    })
    merged.update(kw_overrides)

    # extract positional arguments
    pos_args = sorted(
        (int(k), v) for k, v in merged.items() if isintegral(k)
    )
    if pos_args:
        pos, args = zip(*pos_args)
        if list(pos) != list(range(len(pos))):
            raise ValueError(f"Invalid positions: {pos}")
    else:
        args = ()

    # extract keyword arguments
    kwargs = {k: v for k, v in merged.items() if not isintegral(k)}

    return args, kwargs


def isintegral(txt: str) -> bool:
    try:
        int(txt)
        return True
    except ValueError:
        return False


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

    def __contains__(self, name: object) -> bool:
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

    def clear(self) -> None:
        """Clear configuration."""
        if self._view_path:
            raise RuntimeError("Only root config can be cleared")
        self._config.clear()

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


# default global singleton configuration
config = ConfigView(Config())


# default global singleton function registry
registry: Dict[str, Callable] = {}


def register(
    func_or_class_or_name: Optional[Union[Callable, str]] = None,
    name: Optional[str] = None,
) -> Callable:
    """Register functions/classes as @call config keys."""
    if isinstance(func_or_class_or_name, str):
        # @oh.register("foobar")
        return partial(register, name=func_or_class_or_name)

    if func_or_class_or_name is None:
        # @oh.register
        return partial(register, name=name)

    # @oh.register()
    func = func_or_class_or_name
    name = name or getattr(func, "__name__")

    if not isinstance(name, str) or not str.isidentifier(name):
        raise ValueError(f"Name of callable is not valid: {name}")

    global registry
    registry[name] = func
    return func


def resolve(name: str) -> Callable:
    """Resolve callable by registry or module lookup."""
    global registry
    if not isinstance(name, str):
        raise TypeError(f"Name is not a string: {name}")
    if "/" not in name:
        # registry lookup
        name, attr_path = (name + ".").split(".", 1)
        if name not in registry:
            raise KeyError(f"Name not found in callable registry: {name}")
        obj = registry[name]
        if attr_path:
            obj = nested_getattr(obj, attr_path)
        return obj
    else:
        # module import
        if name.count("/") > 1:
            raise ParseError(f"Too many slashes: {name}")
        module_name, attr_path = name.split("/", 1)
        module = importlib.import_module(module_name)
        return nested_getattr(module, attr_path)


def nested_getattr(obj, path):
    for name in path.split("."):
        obj = getattr(obj, name)
    return obj


def dispatch(func: Callable, args: Tuple, kwargs: Dict) -> Any:
    """Make function call with dynamic arguments."""
    try:
        return func(*args, **kwargs)
    except Exception:
        # generate more helpful error message
        args_txt = ", ".join(repr(a) for a in args)
        if kwargs:
            if args:
                args_txt += ", "
            args_txt += ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
        func_name = getattr(func, "__name__", "<unnamed callable>")
        raise RuntimeError(f"Dispatch failed: {func_name}({args_txt})")
