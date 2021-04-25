
from io import StringIO
import importlib
import json
import re
from functools import partial
from pathlib import Path

from typing import Any, Callable, Dict, Iterator, Optional, List, TextIO, Tuple, Union
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

    @property
    def flat(self) -> Dict[str, Any]:
        """Convenience access to nested config values."""
        def walk(dct: Dict, key: str) -> Any:
            if "." in key:
                first, rest = key.split(".", 1)
                return walk(dct[first], rest)
            else:
                return dct[key]

        def search(dct: Dict) -> Iterator[str]:
            for key, value in dct.items():
                if isinstance(value, dict):
                    for subkey in search(value):
                        yield f"{key}.{subkey}"
                else:
                    yield key

        class FlatConfig(Dict[str, Any]):
            def __contains__(_, key: object) -> bool:
                try:
                    walk(self, str(key))
                    return True
                except KeyError:
                    return False

            def __getitem__(_, key: str) -> Any:
                return walk(self, key)

            def __iter__(_) -> Iterator[str]:
                return search(self)

            def __len__(_) -> int:
                return sum(1 for _ in search(self))

            def __repr__(_) -> str:
                keys = list(search(self))
                return f"FlatConfig({', '.join(map(repr, keys))})"
        return FlatConfig()

    @classmethod
    def from_str(cls, txt: str, *, interpolate: bool = True) -> "Config":
        """Load configuration from string."""
        parser = ConfigParser(
            interpolation=None,
            delimiters=["="],
            comment_prefixes=["#"],
            inline_comment_prefixes=["#"],
            strict=True,
            empty_lines_in_values=False,
        )
        parser.optionxform = str  # type: ignore
        parser.read_string(txt)
        config = Config()
        config._update(parser, interpolate=interpolate)
        return config

    @classmethod
    def from_file(cls, fd: Union[str, Path, TextIO]) -> "Config":
        """Load configuration from file."""
        if isinstance(fd, (str, Path)):
            fd = open(fd)
        return Config.from_str(fd.read())

    @classmethod
    def from_json(cls, data: str) -> "Config":
        """Load configuration from JSON string."""
        parsed = json.loads(data)
        return Config(parsed)

    def to_str(self) -> str:
        """Write the config to a string."""
        writer = ConfigParser()
        for path in self.flat:
            if "." not in path:
                raise SaveError(f"section missing from {repr(path)}")
            section, key = path.rsplit(".", 1)
            value = self.flat[path]
            if key == "@ref":
                # restore interpolation syntax
                if "." not in section:
                    raise SaveError(f"section missing from {repr(section)}")
                section, key = section.rsplit(".", 1)
                if key.startswith("@"):
                    raise SaveError(f"illegal key at {path}: {repr(key)}")
                str_value = "${" + str(value) + "}"
            elif key.startswith("@") or isintegral(key):
                # special @ key, values are unquoted strings
                # integral key, values are integers
                str_value = str(value)
            else:
                str_value = json.dumps(value)
            if not writer.has_section(section):
                writer.add_section(section)
            writer.set(section, key, str_value)
        buf = StringIO()
        writer.write(buf)
        return buf.getvalue().strip()

    def _update(self, parser: ConfigParser, *, interpolate: bool = True) -> None:
        if parser.defaults():
            raise ParseError("Found config values outside of any section")
        json_parser = InterpolatingJSONDecoder(self.flat, interpolate=interpolate)
        for section, values in parser.items():
            if section == "DEFAULT":
                continue
            parts = section.split(".")
            node = self
            for part in parts[:-1]:
                if part not in node:
                    node = node.setdefault(part, Config())
                else:
                    node = node[part]
            if not isinstance(node, dict):
                raise ParseError(f"Found conflicting values for {parts}")
            # Set the default section
            node = node.setdefault(parts[-1], Config())
            for key, value in values.items():
                # parse key
                if key.startswith("@"):
                    # special @ key, values are plain unquoted strings
                    node[key] = str(value)
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
                try:
                    parsed_value = json_parser.decode(value)
                except json.JSONDecodeError as err:
                    raise ParseError(
                        f"Error parsing value of {key}: {repr(value)}: {err}"
                    ) from None
                node[key] = parsed_value


class ParseError(ValueError):
    pass


class SaveError(ValueError):
    pass


class InterpolatingJSONDecoder(json.JSONDecoder):
    """JSON decoder with variable substitution/interpolation."""

    def __init__(self, variables: Dict, *, interpolate: bool = True):
        super().__init__()

        default_parse_array: Callable = self.parse_array
        default_parse_object: Callable = self.parse_object
        default_parse_string: Callable = self.parse_string
        variable_re = re.compile(r"\$\{([^}]+)\}")

        def scan_once(string, idx):
            """Parse value with interpolation."""
            match = variable_re.match(string[idx:])
            if match:
                key = match.groups()[0]
                if interpolate:
                    value = substitute(key, match.string)
                else:
                    value = {"@ref": key}
                return value, idx + match.end()
            else:
                return default_scanner(string, idx)

        def parse_array(s_and_end, _default_scan_once, *args, **kwargs):
            return default_parse_array(s_and_end, scan_once, *args, **kwargs)

        def parse_object(s_and_end, strict, _default_scan_once, *args, **kwargs):
            return default_parse_object(s_and_end, strict, scan_once, *args, **kwargs)

        def parse_string(string, end, strict):
            """Parse JSON string with interpolation.
            Only scalar values can be used in string interpolation.
            """
            string, end = default_parse_string(string, end, strict)
            if not interpolate:
                return string, end
            interpolated = ""
            pos = 0
            for match in variable_re.finditer(string):
                key = match.groups()[0]
                value = substitute(key, match.string)
                if not isinstance(value, (bool, int, float, str, type(None))):
                    # refuse to interpolate nested values in strings
                    raise ParseError(
                        f'String interpolation "{string}" contains '
                        f'non-scalar variable {key}: {value}'
                    )
                var_start, var_end = match.span()
                interpolated += string[pos:var_start] + str(value)
                pos = var_end
            interpolated += string[pos:]
            return interpolated, end

        def substitute(key, var_text):
            try:
                return variables[key]
            except KeyError:
                raise ParseError(f"Variable not found: {repr(var_text)}") from None

        self.parse_array = parse_array
        self.parse_object = parse_object
        self.parse_string = parse_string
        self.scan_once = scan_once
        # must call in the end (refers to self.parse_*)
        default_scanner = json.scanner.py_make_scanner(self)


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
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"Configuration value not found: {name}")

    def __getitem__(self, name: str) -> Any:
        return self._node[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._node)

    def __len__(self) -> int:
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

    def load_str(self, txt: str) -> None:
        """Load configuration from string."""
        if self._view_path:
            raise RuntimeError("Only root config can be loaded")
        config = Config.from_str(txt)
        self._config.update(config)

    def load_file(self, fd: Union[str, Path, TextIO]) -> None:
        if self._view_path:
            raise RuntimeError("Only root config can be loaded")
        config = Config.from_file(fd)
        self._config.update(config)

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
