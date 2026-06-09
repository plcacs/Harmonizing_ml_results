# === Third-party dependency: datadog.dogstatsd ===
# Used symbols: DogStatsd

# === Internal dependency: faust.exceptions ===
class ImproperlyConfigured(FaustError): ...

# === Internal dependency: faust.sensors.monitor ===
class Monitor(Sensor, KeywordReduce):
    ...

# === Third-party dependency: mode.utils.objects ===
class cached_property(Generic[RT]): ...