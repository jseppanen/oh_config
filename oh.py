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
import inspect
import json
import numbers
import re
from configparser import ConfigParser
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from io import StringIO
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    TextIO,
    Tuple,
    Union,
)

try:
    # numpy is an optional dependency
    from numpy import bool_ as np_bool
    from numpy import ndarray as np_ndarray
except ImportError:
    np_bool = bool
    np_ndarray = list


# ints are not strictly json but they're important & supported by Python's json module
JsonData = Union[None, bool, int, float, str, List["JsonData"], Dict[str, "JsonData"]]
JsonSchema = Union[str, List["JsonSchema"], Dict[str, "JsonSchema"]]


class ConfigDict(dict):
    """Dictionary for nested configuration.

    Supports convenient attribute access and callable sections.
    Values are converted to JSON compatible types (plus ints).
    This class validates the types of configuration values against their schema.
    The schema is inferred from the initial configuration entries, and it can
    later be appended to, but not changed.
    """

    def __init__(self, data: Optional[Union[Mapping, Iterable]] = None) -> None:
        """Initialize ConfigDict from mapping or sequence data.

        Values are converted to JSON compatible types (plus ints). Nested dicts are
        converted to ConfigDicts recursively, to implement ConfigDict functionality.
        """
        super().__init__()
        # circuit breaker for infinite recursion from ConfigDict._new(...)
        if data is not None:
            self.update(data)

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
        return func(*args, **kwargs)

    def __getattr__(self, name: str) -> JsonData:
        """Convenient attribute access to dictionary values."""
        if name not in self:
            raise AttributeError(f"ConfigDict has no key {repr(name)}")
        return self[name]

    def __setattr__(self, name: str, value: Any) -> None:
        """Convenient attribute access to dictionary values."""
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __delattr__(self, name: str) -> None:
        raise ValidationError("ConfigDict keys are immutable")

    def __getitem__(self, key: Union[int, str]) -> JsonData:
        """Get configuration values.
        Integer string keys can be accessed by both integer and string types.
        """
        key = cast_as_json_key(key)
        return super().__getitem__(key)

    def __setitem__(self, key: Union[int, str], value: Any) -> None:
        """Set configuration values, with casting to JSON compatible types.
        Casts numpy scalar types to standard (JSON compatible) types.
        Casts mapping and sequence types to (JSON compatible) dicts and lists.
        Validates the resulting types against the built-in configuration schema.
        """
        key = cast_as_json_key(key)
        if key not in self:
            raise ValidationError(f"invalid key: {repr(key)}")
        value = cast_as_json(value, object_hook=ConfigDict._new)
        validate(self[key], value)
        super().__setitem__(key, value)

    def __repr__(self) -> str:
        return f"ConfigDict({repr(self.to_dict())})"

    def __delitem__(self, key: Any) -> None:
        raise ValidationError("ConfigDict keys are immutable")

    def __reduce_ex__(self, protocol):
        """Pickling ConfigDicts converts them to plain old dicts."""
        return dict, (self.to_dict(),)

    @classmethod
    def _new(cls, data: dict) -> "ConfigDict":
        """Create ConfigDict without recursive conversions.

        For use with cast_as_json(..., object_hook=ConfigDict._new)
        """
        res = cls()
        super(cls, res).update(data)
        return res

    def to_dict(self) -> dict:
        """Convert the config into a standard (nested) dict.

        Returns a copy of config using only standard Python types.
        Useful for persistence/pickling.
        """
        # convert nested ConfigDict's into dicts recursively
        return cast_as_json(self)

    def setdefault(
        self, key: Union[int, str], default: Optional[Any] = None
    ) -> JsonData:
        key = cast_as_json_key(key)
        if key not in self:
            raise ValidationError(f"invalid key: {repr(key)}")
        return self[key]

    def pop(self, key, default=None) -> JsonData:
        raise ValidationError("ConfigDict keys are immutable")

    def popitem(self) -> Tuple[str, JsonData]:
        raise ValidationError("ConfigDict keys are immutable")

    def clear(self) -> None:
        raise ValidationError("ConfigDict keys are immutable")

    def update(
        self,
        other: Optional[Union[Mapping, Iterable]] = None,
        *,
        merge_schema: bool = False,
        **kwargs,
    ) -> None:
        """Update config with values from other dict(s) or key-value lists.
        The types of existing keys cannot be changed, but new keys can be added
        with `merge_schema = True`.
        """

        def kvs():
            # dict.update(.) compatible argument handling
            if other:
                if hasattr(other, "keys"):
                    for key in other.keys():
                        yield key, other[key]
                else:
                    for key, value in other:
                        yield key, value
            for key, value in kwargs.items():
                yield key, value

        if not self:
            # initial unvalidated update
            dict_data = dict(kvs())
            # convert nested dicts to ConfigDict's recursively
            data: ConfigDict = cast_as_json(dict_data, object_hook=ConfigDict._new)
            super().update(data)

        else:
            for key, value in kvs():
                key = cast_as_json_key(key)
                if key in self:
                    if isinstance(self[key], ConfigDict):
                        if not isinstance(value, dict):
                            raise ValueError(
                                f"Found conflicting values for {repr(key)}: {repr(value)}"
                            )
                        self[key].update(value, merge_schema=merge_schema)
                    else:
                        self[key] = value
                elif merge_schema:
                    # add new keys without validation
                    value = cast_as_json(value, object_hook=ConfigDict._new)
                    super().__setitem__(key, value)
                else:
                    raise ValidationError(f"invalid key: {repr(key)}")


class Config(ConfigDict):
    """Tree of configuration values.

    This class validates the types of configuration values against their schema.
    The schema is inferred from the initial configuration entries, and it can
    later be appended to, but not changed.
    """

    def __init__(self, data: Optional[dict] = None) -> None:
        super().__init__(data)

    def __repr__(self) -> str:
        return f"Config({repr(self.to_dict())})"

    @property
    def flat(self) -> Mapping[str, JsonData]:
        """Convenience read-only access to nested config values."""
        return FlatDictProxy(self)

    @classmethod
    def from_str(
        cls, txt: str, *, interpolate: bool = True, fill_defaults: bool = True
    ) -> "Config":
        """Load configuration from string."""
        parsed = parse_config(txt, interpolate=interpolate)
        config = cls(parsed)
        if fill_defaults:
            _fill_defaults(config)
        return config

    @classmethod
    def from_file(
        cls,
        fd: Union[str, Path, TextIO],
        *,
        interpolate: bool = True,
        fill_defaults: bool = True,
    ) -> "Config":
        """Load configuration from file."""
        if isinstance(fd, (str, Path)):
            fd = open(fd)
        txt = fd.read()
        parsed = parse_config(txt, interpolate=interpolate)
        config = cls(parsed)
        if fill_defaults:
            _fill_defaults(config)
        return config

    @classmethod
    def from_json(cls, data: str) -> "Config":
        """Load configuration from JSON string."""
        parsed = json.loads(data)
        return cls(parsed)

    def load_str(
        self, txt: str, *, interpolate: bool = True, merge_schema: bool = False
    ) -> None:
        """Load configuration from string."""
        config = Config.from_str(txt, interpolate=interpolate)
        self.update(config, merge_schema=merge_schema)

    def load_file(
        self,
        fd: Union[str, Path, TextIO],
        *,
        interpolate: bool = True,
        merge_schema: bool = False,
    ) -> None:
        """Load configuration from file."""
        config = Config.from_file(fd, interpolate=interpolate)
        self.update(config, merge_schema=merge_schema)

    def load_json(self, data: str) -> None:
        """Load configuration from JSON string."""
        parsed = json.loads(data)
        config = Config(parsed)
        self.update(config)

    def to_str(self) -> str:
        """Write the config to a string."""
        return dump_config(self)


class FlatDictProxy(Mapping[str, JsonData]):
    """Immutable proxy for accessing nested dictionary values as flat dictionary."""

    def __init__(self, dct: dict) -> None:
        self.dct = dct

    def __contains__(self, key: object) -> bool:
        try:
            self[str(key)]
            return True
        except KeyError:
            return False

    def __getitem__(self, key: str) -> JsonData:
        def walk(dct: Mapping, key: str) -> JsonData:
            if "." in key:
                first, rest = key.split(".", 1)
                return walk(dct[first], rest)
            else:
                return dct[key]

        return walk(self.dct, key)

    def __iter__(self) -> Iterator[str]:
        def search(dct: Mapping) -> Iterator[str]:
            for key, value in dct.items():
                if isinstance(value, dict):
                    for subkey in search(value):
                        yield f"{key}.{subkey}"
                else:
                    yield key

        return search(self.dct)

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))

    def __repr__(self) -> str:
        keys = list(self)
        return f"FlatDictProxy({', '.join(map(repr, keys))})"


def parse_config(txt: str, *, interpolate: bool = True) -> Dict[str, Any]:
    """Parse configuration from string."""

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
    if parser.defaults():
        raise ParseError("Found config values outside of any section")

    result: Dict[str, Any] = {}
    json_parser = InterpolatingJSONDecoder(
        FlatDictProxy(result) if interpolate else None
    )

    for section, values in parser.items():
        if section == "DEFAULT":
            continue
        parts = section.split(".")
        node = result
        for part in parts[:-1]:
            if part not in node:
                node = node.setdefault(part, {})
            else:
                node = node[part]
        if not isinstance(node, dict):
            raise ParseError(f"Found conflicting values for {parts}")
        # Set the default section
        node = node.setdefault(parts[-1], {})
        for key, value in values.items():
            # parse key
            if key.startswith("@"):
                # special @ key, values are plain unquoted strings
                node[key] = str(value)
                continue
            elif isintegral_str(key):
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
    return result


def dump_config(conf: Dict[str, Any]) -> str:
    """Dump configuration as string."""

    flat_conf = FlatDictProxy(conf)
    writer = ConfigParser()
    for path in flat_conf:
        if "." not in path:
            raise SaveError(f"section missing from {repr(path)}")
        section, key = path.rsplit(".", 1)
        value = flat_conf[path]
        if key == "@ref":
            # restore interpolation syntax
            if "." not in section:
                raise SaveError(f"section missing from {repr(section)}")
            section, key = section.rsplit(".", 1)
            if key.startswith("@"):
                raise SaveError(f"illegal key at {path}: {repr(key)}")
            str_value = "${" + str(value) + "}"
        elif key.startswith("@") or isintegral_str(key):
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


def cast_as_json_key(key: Union[int, str]) -> str:
    """Cast config keys to JSON compatible strings."""

    if isint(key):
        # integral keys are used for positional arguments
        if key < 0:
            raise ValueError(f"Negative integer keys are not valid: {key}")
        key = str(key)
    if not isinstance(key, str):
        raise TypeError(f"not JSON compatible: key is not string: {repr(key)}")
    if not (str.isidentifier(key) or isintegral_str(key) or key.startswith("@")):
        raise ParseError(f"Key is not valid: {repr(key)}")
    return key


def cast_as_json(value: Any, *, object_hook=None) -> JsonData:
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
        return [cast_as_json(v, object_hook=object_hook) for v in value]
    elif isinstance(value, dict):
        value = dict(
            (cast_as_json_key(k), cast_as_json(v, object_hook=object_hook))
            for k, v in value.items()
        )
        if object_hook:
            return object_hook(value)
        return value
    else:
        raise TypeError(f"not JSON compatible: {type(value).__name__}: {repr(value)}")


def validate(reference: JsonData, value: JsonData) -> None:
    """Validate config values against the corresponding schema."""

    schema = infer_schema(reference)
    if infer_schema(value) != schema:
        raise ValidationError(
            f"invalid value {repr(value)} for schema {schema}; "
            f"inferred as {infer_schema(value)}"
        )


def infer_schema(data: JsonData) -> JsonSchema:
    if data is None:
        return "None"
    elif isinstance(data, (bool, int, float, str)):
        return type(data).__name__
    elif isinstance(data, list):
        if data == []:
            # cannot validate empty list by anything other than an empty list :-/
            return []
        else:
            # intended for homogeneous arrays/tensors
            return [infer_schema(data[0])]
    elif isinstance(data, dict):
        return {k: infer_schema(v) for k, v in data.items()}
    else:
        raise TypeError(f"not JSON data: {repr(data)}")


class ParseError(ValueError):
    pass


class SaveError(ValueError):
    pass


class ValidationError(ValueError):
    pass


class InterpolatingJSONDecoder(json.JSONDecoder):
    """JSON decoder with variable substitution/interpolation."""

    def __init__(self, variables: Optional[Mapping] = None) -> None:
        super().__init__()
        interpolate = variables is not None

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
    pos_args = sorted((int(k), v) for k, v in merged.items() if isintegral_str(k))
    if pos_args:
        pos, args = zip(*pos_args)
        if list(pos) != list(range(len(pos))):
            raise ValueError(f"Invalid positions: {pos}")
    else:
        args = ()

    # extract keyword arguments
    kwargs = {k: v for k, v in merged.items() if not isintegral_str(k)}

    return args, kwargs


def isint(x: Any) -> bool:
    """Return true iff argument is an integer."""
    return isinstance(x, int) and not isinstance(x, bool)


def isintegral_str(txt: str) -> bool:
    """Return true iff argument is an integer string."""
    if not isinstance(txt, str):
        return False
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
    def _node(self) -> ConfigDict:
        node = self._config
        for key in self._view_path:
            node = node[key]
        return node

    def reset(self) -> None:
        """Clear configuration."""
        if self._view_path:
            raise RuntimeError("Only root config can be cleared")
        self._config = Config()

    def load_str(self, txt: str, **kwargs) -> None:
        """Load configuration from string."""
        if self._view_path:
            raise RuntimeError("Only root config can be loaded")
        self._config.load_str(txt, **kwargs)

    def load_file(self, fd: Union[str, Path, TextIO], **kwargs) -> None:
        if self._view_path:
            raise RuntimeError("Only root config can be loaded")
        self._config.load_file(fd, **kwargs)

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
            elif not isinstance(self._node[part], ConfigDict):
                raise KeyError(f"Not a section: {part}: {self._node[part]}")
            else:
                self._view_path.append(part)
        try:
            yield
        finally:
            for part in parts[::-1]:
                popped = self._view_path.pop()
                assert popped == part, "push/pop invariant failed"


@dataclass
class ConfigFunction:
    func: Callable
    with_defaults: bool = False

    def __call__(self, *args, **kwargs) -> Any:
        try:
            return self.func(*args, **kwargs)
        except Exception as err:
            # generate more helpful error message
            args_txt = ", ".join(repr(a) for a in args)
            if kwargs:
                if args:
                    args_txt += ", "
                args_txt += ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
            func_name = getattr(self.func, "__name__", "<unnamed callable>")
            msg = (
                f"Call failed: {func_name}({args_txt}): {err.__class__.__name__}: {err}"
            )
            raise RuntimeError(msg) from err

    def get_defaults(self) -> Dict[str, Any]:
        return _inspect_defaults(self.func) if self.with_defaults else {}


# default global singleton configuration
config = ConfigView(Config())


# default global singleton function registry
registry: Dict[str, ConfigFunction] = {}


def register(
    func_or_class_or_name: Optional[Union[Callable, str]] = None,
    name: Optional[str] = None,
    *,
    with_defaults: bool = False,
) -> Callable:
    """Register functions/classes as @call config keys.
    :param with_defaults: Parse default configuration from function signature.
    """
    if isinstance(func_or_class_or_name, str):
        # @oh.register("foobar")
        return partial(
            register, name=func_or_class_or_name, with_defaults=with_defaults
        )

    if func_or_class_or_name is None:
        # @oh.register
        return partial(register, name=name, with_defaults=with_defaults)

    # @oh.register()
    func = func_or_class_or_name
    name = name or getattr(func, "__name__")

    if not isinstance(name, str) or not str.isidentifier(name):
        raise ValueError(f"Name of callable is not valid: {name}")

    global registry
    registry[name] = ConfigFunction(func, with_defaults)
    return func


def _inspect_defaults(func: Callable) -> Dict[str, Any]:
    signature = inspect.signature(func)
    defaults = {
        name: param.default
        for name, param in signature.parameters.items()
        if param.default != inspect.Parameter.empty
    }
    return defaults


def _fill_defaults(config: ConfigDict) -> None:
    """Traverse config for @call sections and fill their default values."""

    if "@call" in config:
        func = resolve(config["@call"])
        defaults = func.get_defaults()
        config.update(
            (
                (name, default)
                for pos, (name, default) in enumerate(defaults.items())
                if name not in config and str(pos) not in config
            ),
            merge_schema=True,
        )
    else:
        for child in config.values():
            if isinstance(child, ConfigDict):
                _fill_defaults(child)


def resolve(name: str) -> ConfigFunction:
    """Resolve callable by registry or module lookup."""
    global registry
    if not isinstance(name, str):
        raise TypeError(f"Name is not a string: {name}")
    if ":" not in name:
        # registry lookup
        if name not in registry:
            raise KeyError(f"Name not found in callable registry: {name}")
        return registry[name]
    else:
        # module import
        if name.count(":") > 1:
            raise ParseError(f"Expected <module>:<function>, got: {name}")
        module_name, attr_path = name.split(":", 1)
        module = importlib.import_module(module_name)
        func = nested_getattr(module, attr_path)
        return ConfigFunction(func)


def nested_getattr(obj, path):
    for name in path.split("."):
        obj = getattr(obj, name)
    return obj
