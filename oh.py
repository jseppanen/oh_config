"""
The 0 (oh) config module provides simple configuration for machine learning
experiments.

Some highlights:
* Friendly TOML/INI style configuration format
* JSON compatible
* Configuration is a tree of plain old data
* Configuration is not executable
* No need to string configuration values through the call stack
* Single source file, easy to drop in to any project
* Zero dependencies
* No command line interface
* Not coupled with any software framework or paradigm
"""

# MIT License
#
# Copyright (C) 2016 ExplosionAI GmbH, 2016 spaCy GmbH, 2015 Matthew Honnibal
# Copyright (c) 2021 Jarno Seppänen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import importlib
import json
import numbers
import re
from configparser import ConfigParser
from contextlib import contextmanager
from functools import partial
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, TextIO, Tuple, Union

try:
    # numpy is an optional dependency
    from numpy import bool_ as np_bool
    from numpy import ndarray as np_ndarray
except ImportError:
    np_bool = bool
    np_ndarray = list


# ints are not strictly json but they're important & supported by Python's json module
JsonData = Union[None, bool, int, float, str, List["JsonData"], Dict[str, "JsonData"]]


class ConfigDict(dict):
    """Dictionary for nested configuration.

    Supports convenient attribute access and callable sections.
    Values are converted to JSON compatible types (plus ints).
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

    def __getattr__(self, name: str) -> JsonData:
        """Convenient attribute access to dictionary values."""
        if name not in self:
            raise AttributeError(f"dictionary has no key {repr(name)}")
        return self[name]

    def __setattr__(self, name: str, value: Any) -> None:
        """Convenient attribute access to dictionary values."""
        self[name] = value

    def __delattr__(self, name: str) -> None:
        """Convenient attribute access to dictionary values."""
        del self[name]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration values, with casting to JSON compatible types.
        Casts numpy scalar types to standard (JSON compatible) types.
        Casts mapping and sequence types to (JSON compatible) dicts and lists.
        """
        # support nested attribute access
        value = cast(value, object_hook=ConfigDict)
        super().__setitem__(key, value)


class Config(ConfigDict):
    """Tree of configuration values."""

    @property
    def flat(self) -> Dict[str, JsonData]:
        """Convenience access to nested config values."""

        def walk(dct: Dict, key: str) -> JsonData:
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

        class FlatConfig(Dict[str, JsonData]):
            def __contains__(_, key: object) -> bool:
                try:
                    walk(self, str(key))
                    return True
                except KeyError:
                    return False

            def __getitem__(_, key: str) -> JsonData:
                return walk(self, key)

            def __setitem__(_, key: str, value: Any) -> None:
                raise KeyError("not supported")

            def __delitem__(_, key: str) -> None:
                raise KeyError("not supported")

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
        config = Config()
        config.load_str(txt, interpolate=interpolate)
        return config

    @classmethod
    def from_file(
        cls, fd: Union[str, Path, TextIO], *, interpolate: bool = True
    ) -> "Config":
        """Load configuration from file."""
        config = Config()
        config.load_file(fd, interpolate=interpolate)
        return config

    @classmethod
    def from_json(cls, data: str) -> "Config":
        """Load configuration from JSON string."""
        parsed = json.loads(data)
        return Config(parsed)

    def load_str(self, txt: str, *, interpolate: bool = True) -> None:
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
        self._update(parser, interpolate=interpolate)

    def load_file(
        self, fd: Union[str, Path, TextIO], *, interpolate: bool = True
    ) -> None:
        """Load configuration from file."""
        if isinstance(fd, (str, Path)):
            fd = open(fd)
        self.load_str(fd.read(), interpolate=interpolate)

    def load_json(self, data: str) -> None:
        """Load configuration from JSON string."""
        parsed = json.loads(data)
        self.update(parsed)

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
                    node = node.setdefault(part, ConfigDict())
                else:
                    node = node[part]
            if not isinstance(node, dict):
                raise ParseError(f"Found conflicting values for {parts}")
            # Set the default section
            node = node.setdefault(parts[-1], ConfigDict())
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


def cast(value: Any, *, object_hook=None) -> JsonData:
    """Cast config values to JSON compatible standard types."""

    if value is None:
        return None
    elif isinstance(value, (bool, np_bool)):
        # test bool before Integral
        return bool(value)
    elif isinstance(value, numbers.Integral):
        # test Integral before Real
        return int(value)
    elif isinstance(value, numbers.Real):
        return float(value)
    elif isinstance(value, str):
        return str(value)
    elif isinstance(value, (list, tuple, np_ndarray)):
        return [cast(v, object_hook=object_hook) for v in value]
    elif isinstance(value, dict):

        def check(key):
            if not isinstance(key, str):
                raise TypeError(
                    f"not JSON compatible: key is not string: {repr(key)} in {repr(value)}"
                )
            return key

        value = dict(
            (check(k), cast(v, object_hook=object_hook)) for k, v in value.items()
        )
        if object_hook:
            return object_hook(value)
        return value
    else:
        raise TypeError(f"not JSON compatible: {type(value).__name__}: {repr(value)}")


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
                        f"non-scalar variable {key}: {value}"
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
    merged.update({str(k): v for k, v in enumerate(pos_overrides)})
    merged.update(kw_overrides)

    # extract positional arguments
    pos_args = sorted((int(k), v) for k, v in merged.items() if isintegral(k))
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


class ConfigView(Dict[str, JsonData]):
    """View of a Config (sub)section.
    Used for traversing subsections with a context manager.
    """

    def __init__(self, config: Config):
        self._config = config
        self._view_path: List[str] = []

    def __getattr__(self, name: str) -> JsonData:
        """Convenience attribute access to config values."""
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"Configuration value not found: {name}")

    def __getitem__(self, name: str) -> JsonData:
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
        self._config.load_str(txt)

    def load_file(self, fd: Union[str, Path, TextIO]) -> None:
        if self._view_path:
            raise RuntimeError("Only root config can be loaded")
        self._config.load_file(fd)

    def update(self, *args, **kwargs):
        if self._view_path:
            raise RuntimeError("Only root config can be updated")
        self._config.update(*args, **kwargs)

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
    if ":" not in name:
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
        if name.count(":") > 1:
            raise ParseError(f"Expected <module>:<function>, got: {name}")
        module_name, attr_path = name.split(":", 1)
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
    except Exception as err:
        # generate more helpful error message
        args_txt = ", ".join(repr(a) for a in args)
        if kwargs:
            if args:
                args_txt += ", "
            args_txt += ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
        func_name = getattr(func, "__name__", "<unnamed callable>")
        msg = (
            f"Dispatch failed: {func_name}({args_txt}): {err.__class__.__name__}: {err}"
        )
        raise RuntimeError(msg) from err
