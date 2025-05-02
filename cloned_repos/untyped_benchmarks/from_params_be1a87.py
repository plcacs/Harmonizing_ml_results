import collections.abc
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, cast, Dict, Iterable, List, Mapping, Set, Tuple, Type, TypeVar, Union, Optional
import inspect
import logging
from allennlp.common.checks import ConfigurationError
from allennlp.common.lazy import Lazy
from allennlp.common.params import Params
logger = logging.getLogger(__name__)
T = TypeVar('T', bound='FromParams')
_NO_DEFAULT = inspect.Parameter.empty

def takes_arg(obj, arg):
    """
    Checks whether the provided obj takes a certain arg.
    If it's a class, we're really checking whether its constructor does.
    If it's a function or method, we're checking the object itself.
    Otherwise, we raise an error.
    """
    if inspect.isclass(obj):
        signature = inspect.signature(obj.__init__)
    elif inspect.ismethod(obj) or inspect.isfunction(obj):
        signature = inspect.signature(obj)
    else:
        raise ConfigurationError(f'object {obj} is not callable')
    return arg in signature.parameters

def takes_kwargs(obj):
    """
    Checks whether a provided object takes in any positional arguments.
    Similar to takes_arg, we do this for both the __init__ function of
    the class or a function / method
    Otherwise, we raise an error
    """
    if inspect.isclass(obj):
        signature = inspect.signature(obj.__init__)
    elif inspect.ismethod(obj) or inspect.isfunction(obj):
        signature = inspect.signature(obj)
    else:
        raise ConfigurationError(f'object {obj} is not callable')
    return any((p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()))

def can_construct_from_params(type_):
    if type_ in [str, int, float, bool]:
        return True
    origin = getattr(type_, '__origin__', None)
    if origin == Lazy:
        return True
    elif origin:
        if hasattr(type_, 'from_params'):
            return True
        args = getattr(type_, '__args__')
        return all((can_construct_from_params(arg) for arg in args))
    return hasattr(type_, 'from_params')

def is_base_registrable(cls):
    """
    Checks whether this is a class that directly inherits from Registrable, or is a subclass of such
    a class.
    """
    from allennlp.common.registrable import Registrable
    if not issubclass(cls, Registrable):
        return False
    method_resolution_order = inspect.getmro(cls)[1:]
    for base_class in method_resolution_order:
        if issubclass(base_class, Registrable) and base_class is not Registrable:
            return False
    return True

def remove_optional(annotation):
    """
    Optional[X] annotations are actually represented as Union[X, NoneType].
    For our purposes, the "Optional" part is not interesting, so here we
    throw it away.
    """
    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', ())
    if origin == Union:
        return Union[tuple([arg for arg in args if arg != type(None)])]
    else:
        return annotation

def infer_constructor_params(cls, constructor=None):
    if constructor is None:
        constructor = cls.__init__
    return infer_method_params(cls, constructor)
infer_params = infer_constructor_params

def infer_method_params(cls, method):
    signature = inspect.signature(method)
    parameters = dict(signature.parameters)
    has_kwargs = False
    var_positional_key = None
    for param in parameters.values():
        if param.kind == param.VAR_KEYWORD:
            has_kwargs = True
        elif param.kind == param.VAR_POSITIONAL:
            var_positional_key = param.name
    if var_positional_key:
        del parameters[var_positional_key]
    if not has_kwargs:
        return parameters
    super_class = None
    for super_class_candidate in cls.mro()[1:]:
        if issubclass(super_class_candidate, FromParams):
            super_class = super_class_candidate
            break
    if super_class:
        super_parameters = infer_params(super_class)
    else:
        super_parameters = {}
    return {**super_parameters, **parameters}

def create_kwargs(constructor, cls, params, **extras):
    """
    Given some class, a `Params` object, and potentially other keyword arguments,
    create a dict of keyword args suitable for passing to the class's constructor.

    The function does this by finding the class's constructor, matching the constructor
    arguments to entries in the `params` object, and instantiating values for the parameters
    using the type annotation and possibly a from_params method.

    Any values that are provided in the `extras` will just be used as is.
    For instance, you might provide an existing `Vocabulary` this way.
    """
    kwargs = {}
    parameters = infer_params(cls, constructor)
    accepts_kwargs = False
    for param_name, param in parameters.items():
        if param_name == 'self':
            continue
        if param.kind == param.VAR_KEYWORD:
            accepts_kwargs = True
            continue
        annotation = remove_optional(param.annotation)
        explicitly_set = param_name in params
        constructed_arg = pop_and_construct_arg(cls.__name__, param_name, annotation, param.default, params, **extras)
        if explicitly_set or constructed_arg is not param.default:
            kwargs[param_name] = constructed_arg
    if accepts_kwargs:
        kwargs.update(params)
    else:
        params.assert_empty(cls.__name__)
    return kwargs

def create_extras(cls, extras):
    """
    Given a dictionary of extra arguments, returns a dictionary of
    kwargs that actually are a part of the signature of the cls.from_params
    (or cls) method.
    """
    subextras = {}
    if hasattr(cls, 'from_params'):
        from_params_method = cls.from_params
    else:
        from_params_method = cls
    if takes_kwargs(from_params_method):
        subextras = extras
    else:
        subextras = {k: v for k, v in extras.items() if takes_arg(from_params_method, k)}
    return subextras

def pop_and_construct_arg(class_name, argument_name, annotation, default, params, **extras):
    """
    Does the work of actually constructing an individual argument for
    [`create_kwargs`](./#create_kwargs).

    Here we're in the inner loop of iterating over the parameters to a particular constructor,
    trying to construct just one of them.  The information we get for that parameter is its name,
    its type annotation, and its default value; we also get the full set of `Params` for
    constructing the object (which we may mutate), and any `extras` that the constructor might
    need.

    We take the type annotation and default value here separately, instead of using an
    `inspect.Parameter` object directly, so that we can handle `Union` types using recursion on
    this method, trying the different annotation types in the union in turn.
    """
    from allennlp.models.archival import load_archive
    name = argument_name
    if name in extras:
        if name not in params:
            return extras[name]
        else:
            logger.warning(f"Parameter {name} for class {class_name} was found in both **extras and in params. Using the specification found in params, but you probably put a key in a config file that you didn't need, and if it is different from what we get from **extras, you might get unexpected behavior.")
    elif name in params and isinstance(params.get(name), Params) and ('_pretrained' in params.get(name)):
        load_module_params = params.pop(name).pop('_pretrained')
        archive_file = load_module_params.pop('archive_file')
        module_path = load_module_params.pop('module_path')
        freeze = load_module_params.pop('freeze', True)
        archive = load_archive(archive_file)
        result = archive.extract_module(module_path, freeze)
        if not isinstance(result, annotation):
            raise ConfigurationError(f'The module from model at {archive_file} at path {module_path} was expected of type {annotation} but is of type {type(result)}')
        return result
    popped_params = params.pop(name, default) if default != _NO_DEFAULT else params.pop(name)
    if popped_params is None:
        return None
    return construct_arg(class_name, name, popped_params, annotation, default, **extras)

def construct_arg(class_name, argument_name, popped_params, annotation, default, **extras):
    """
    The first two parameters here are only used for logging if we encounter an error.
    """
    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', [])
    optional = default != _NO_DEFAULT
    if hasattr(annotation, 'from_params'):
        if popped_params is default:
            return default
        elif popped_params is not None:
            subextras = create_extras(annotation, extras)
            if isinstance(popped_params, str):
                popped_params = Params({'type': popped_params})
            elif isinstance(popped_params, dict):
                popped_params = Params(popped_params)
            result = annotation.from_params(params=popped_params, **subextras)
            return result
        elif not optional:
            raise ConfigurationError(f'expected key {argument_name} for {class_name}')
        else:
            return default
    elif annotation in {int, bool}:
        if type(popped_params) in {int, bool}:
            return annotation(popped_params)
        else:
            raise TypeError(f'Expected {argument_name} to be a {annotation.__name__}.')
    elif annotation == str:
        if type(popped_params) == str or isinstance(popped_params, Path):
            return str(popped_params)
        else:
            raise TypeError(f'Expected {argument_name} to be a string.')
    elif annotation == float:
        if type(popped_params) in {int, float}:
            return popped_params
        else:
            raise TypeError(f'Expected {argument_name} to be numeric.')
    elif origin in {collections.abc.Mapping, Mapping, Dict, dict} and len(args) == 2 and can_construct_from_params(args[-1]):
        value_cls = annotation.__args__[-1]
        value_dict = {}
        if not isinstance(popped_params, Mapping):
            raise TypeError(f'Expected {argument_name} to be a Mapping (probably a dict or a Params object).')
        for key, value_params in popped_params.items():
            value_dict[key] = construct_arg(str(value_cls), argument_name + '.' + key, value_params, value_cls, _NO_DEFAULT, **extras)
        return value_dict
    elif origin in (Tuple, tuple) and all((can_construct_from_params(arg) for arg in args)):
        value_list = []
        for i, (value_cls, value_params) in enumerate(zip(annotation.__args__, popped_params)):
            value = construct_arg(str(value_cls), argument_name + f'.{i}', value_params, value_cls, _NO_DEFAULT, **extras)
            value_list.append(value)
        return tuple(value_list)
    elif origin in (Set, set) and len(args) == 1 and can_construct_from_params(args[0]):
        value_cls = annotation.__args__[0]
        value_set = set()
        for i, value_params in enumerate(popped_params):
            value = construct_arg(str(value_cls), argument_name + f'.{i}', value_params, value_cls, _NO_DEFAULT, **extras)
            value_set.add(value)
        return value_set
    elif origin == Union:
        backup_params = deepcopy(popped_params)
        error_chain = None
        for arg_annotation in args:
            try:
                return construct_arg(str(arg_annotation), argument_name, popped_params, arg_annotation, default, **extras)
            except (ValueError, TypeError, ConfigurationError, AttributeError) as e:
                popped_params = deepcopy(backup_params)
                e.args = (f'While constructing an argument of type {arg_annotation}',) + e.args
                e.__cause__ = error_chain
                error_chain = e
        config_error = ConfigurationError(f'Failed to construct argument {argument_name} with type {annotation}.')
        config_error.__cause__ = error_chain
        raise config_error
    elif origin == Lazy:
        if popped_params is default:
            return default
        value_cls = args[0]
        subextras = create_extras(value_cls, extras)
        return Lazy(value_cls, params=deepcopy(popped_params), constructor_extras=subextras)
    elif origin in {collections.abc.Iterable, Iterable, List, list} and len(args) == 1 and can_construct_from_params(args[0]):
        value_cls = annotation.__args__[0]
        value_list = []
        for i, value_params in enumerate(popped_params):
            value = construct_arg(str(value_cls), argument_name + f'.{i}', value_params, value_cls, _NO_DEFAULT, **extras)
            value_list.append(value)
        return value_list
    else:
        if isinstance(popped_params, Params):
            return popped_params.as_dict()
        return popped_params

class FromParams:
    """
    Mixin to give a from_params method to classes. We create a distinct base class for this
    because sometimes we want non-Registrable classes to be instantiatable from_params.
    """

    @classmethod
    def from_params(cls, params, constructor_to_call=None, constructor_to_inspect=None, **extras):
        """
        This is the automatic implementation of `from_params`. Any class that subclasses
        `FromParams` (or `Registrable`, which itself subclasses `FromParams`) gets this
        implementation for free.  If you want your class to be instantiated from params in the
        "obvious" way -- pop off parameters and hand them to your constructor with the same names --
        this provides that functionality.

        If you need more complex logic in your from `from_params` method, you'll have to implement
        your own method that overrides this one.

        The `constructor_to_call` and `constructor_to_inspect` arguments deal with a bit of
        redirection that we do.  We allow you to register particular `@classmethods` on a class as
        the constructor to use for a registered name.  This lets you, e.g., have a single
        `Vocabulary` class that can be constructed in two different ways, with different names
        registered to each constructor.  In order to handle this, we need to know not just the class
        we're trying to construct (`cls`), but also what method we should inspect to find its
        arguments (`constructor_to_inspect`), and what method to call when we're done constructing
        arguments (`constructor_to_call`).  These two methods are the same when you've used a
        `@classmethod` as your constructor, but they are `different` when you use the default
        constructor (because you inspect `__init__`, but call `cls()`).
        """
        from allennlp.common.registrable import Registrable
        logger.debug(f'instantiating class {cls} from params {getattr(params, 'params', params)} and extras {set(extras.keys())}')
        if params is None:
            return None
        if isinstance(params, str):
            params = Params({'type': params})
        if not isinstance(params, Params):
            raise ConfigurationError(f'from_params was passed a `params` object that was not a `Params`. This probably indicates malformed parameters in a configuration file, where something that should have been a dictionary was actually a list, or something else. This happened when constructing an object of type {cls}.')
        registered_subclasses = Registrable._registry.get(cls)
        if is_base_registrable(cls) and registered_subclasses is None:
            raise ConfigurationError('Tried to construct an abstract Registrable base class that has no registered concrete types. This might mean that you need to use --include-package to get your concrete classes actually registered.')
        if registered_subclasses is not None and (not constructor_to_call):
            as_registrable = cast(Type[Registrable], cls)
            default_to_first_choice = as_registrable.default_implementation is not None
            choice = params.pop_choice('type', choices=as_registrable.list_available(), default_to_first_choice=default_to_first_choice)
            subclass, constructor_name = as_registrable.resolve_class_name(choice)
            if not constructor_name:
                constructor_to_inspect = subclass.__init__
                constructor_to_call = subclass
            else:
                constructor_to_inspect = cast(Callable[..., T], getattr(subclass, constructor_name))
                constructor_to_call = constructor_to_inspect
            if hasattr(subclass, 'from_params'):
                extras = create_extras(subclass, extras)
                retyped_subclass = cast(Type[T], subclass)
                return retyped_subclass.from_params(params=params, constructor_to_call=constructor_to_call, constructor_to_inspect=constructor_to_inspect, **extras)
            else:
                return subclass(**params)
        else:
            if not constructor_to_inspect:
                constructor_to_inspect = cls.__init__
            if not constructor_to_call:
                constructor_to_call = cls
            if constructor_to_inspect == object.__init__:
                kwargs = {}
                params.assert_empty(cls.__name__)
            else:
                constructor_to_inspect = cast(Callable[..., T], constructor_to_inspect)
                kwargs = create_kwargs(constructor_to_inspect, cls, params, **extras)
            return constructor_to_call(**kwargs)

    def to_params(self):
        """
        Returns a `Params` object that can be used with `.from_params()` to recreate an
        object just like it.

        This relies on `_to_params()`. If you need this in your custom `FromParams` class,
        override `_to_params()`, not this method.
        """

        def replace_object_with_params(o):
            if isinstance(o, FromParams):
                return o.to_params()
            elif isinstance(o, List):
                return [replace_object_with_params(i) for i in o]
            elif isinstance(o, Set):
                return {replace_object_with_params(i) for i in o}
            elif isinstance(o, Dict):
                return {key: replace_object_with_params(value) for key, value in o.items()}
            else:
                return o
        return Params(replace_object_with_params(self._to_params()))

    def _to_params(self):
        """
        Returns a dictionary of parameters that, when turned into a `Params` object and
        then fed to `.from_params()`, will recreate this object.

        You don't need to implement this all the time. AllenNLP will let you know if you
        need it.
        """
        raise NotImplementedError()