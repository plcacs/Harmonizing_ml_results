# === Internal dependency: allennlp.confidence_checks.task_checklists.task_suite ===
class TaskSuite(Registrable):
    ...

# === Internal dependency: allennlp.confidence_checks.task_checklists.utils ===
def spacy_wrap(fn: Callable, language: str = ..., **kwargs) -> Callable: ...
def strip_punctuation(data: Union[str, spacy.tokens.doc.Doc]) -> str: ...
def add_random_strings(data: str) -> List[str]: ...

# === Internal dependency: allennlp.data.instance ===
class Instance(Mapping[str, Field]): ...

# === Unresolved dependency: checklist.perturb ===
# Used unresolved symbols: Perturb

# === Unresolved dependency: checklist.test_types ===
# Used unresolved symbols: DIR, Expect, INV, MFT

# === Third-party dependency: numpy ===
# Used symbols: array, random