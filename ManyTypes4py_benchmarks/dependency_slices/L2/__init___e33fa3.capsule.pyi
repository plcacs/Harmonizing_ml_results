from typing import Any

# === Third-party dependency: jedi ===
# Used symbols: InterpreterEnvironment, debug

# === Third-party dependency: jedi._compatibility ===
def pickle_load(file) -> Any: ...
def pickle_dump(data, file, protocol) -> Any: ...

# === Third-party dependency: jedi.api.exceptions ===
class InternalError(_JediError): ...

# === Third-party dependency: jedi.cache ===
def memoize_method(method) -> Any: ...

# === Third-party dependency: jedi.inference ===
class InferenceState:
    def __init__(self, project, environment = ..., script_path = ...) -> Any: ...

# === Third-party dependency: jedi.inference.compiled.access ===
class AccessPath: ...
class DirectObjectAccess:
    def __init__(self, inference_state, obj) -> Any: ...
SignatureParam: namedtuple

# === Third-party dependency: jedi.inference.compiled.subprocess ===
# Used symbols: functions