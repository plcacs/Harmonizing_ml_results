# === Internal dependency: allennlp.confidence_checks.task_checklists.task_suite ===
class TaskSuite(Registrable):
    ...

# === Internal dependency: allennlp.confidence_checks.task_checklists.utils ===
def toggle_punctuation(data: str) -> List[str]: ...
def add_random_strings(data: str) -> List[str]: ...

# === Unresolved dependency: checklist.perturb ===
# Used unresolved symbols: Perturb

# === Unresolved dependency: checklist.test_types ===
# Used unresolved symbols: DIR, Expect, INV, MFT

# === Third-party dependency: numpy ===
# Used symbols: argmax, array