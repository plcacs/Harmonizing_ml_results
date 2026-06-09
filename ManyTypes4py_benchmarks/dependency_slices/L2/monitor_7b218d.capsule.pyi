# === Internal dependency: faust.sensors.base ===
class Sensor(SensorT, Service):
    ...

# === Internal dependency: faust.types ===
# re-export: from .tables import CollectionT

# === Internal dependency: faust.types.tuples ===
class TP(NamedTuple): ...

# === Internal dependency: faust.utils.functional ===
def deque_pushpopmax(l: Deque[T], item: T, max: int = ...) -> Optional[T]: ...

# === Unresolved dependency: mode ===
# Used unresolved symbols: Service, label

# === Third-party dependency: mode.utils.objects ===
class KeywordReduce:
    ...

# === Third-party dependency: mode.utils.typing ===
# Used symbols: Counter, Deque