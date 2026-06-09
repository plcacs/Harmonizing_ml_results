from typing import Any

# === Internal dependency: allennlp.common.file_utils ===
def cached_path(url_or_filename: Union[str, PathLike], cache_dir: Union[str, Path] = ..., extract_archive: bool = ..., force_extract: bool = ...) -> str: ...

# === Internal dependency: allennlp.common.registrable ===
class Registrable(FromParams):
    ...

# === Internal dependency: allennlp.confidence_checks.task_checklists.utils ===
def add_common_lexicons(editor: Editor) -> Any: ...
def toggle_punctuation(data: str) -> List[str]: ...

# === Unresolved dependency: checklist.editor ===
# Used unresolved symbols: Editor

# === Unresolved dependency: checklist.perturb ===
# Used unresolved symbols: Perturb

# === Unresolved dependency: checklist.test_suite ===
# Used unresolved symbols: TestSuite

# === Unresolved dependency: checklist.test_types ===
# Used unresolved symbols: INV