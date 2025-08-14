def is_absolute(path: str) -> bool:
    return any(path.startswith(x) for x in ["/", "http:", "https:"])
