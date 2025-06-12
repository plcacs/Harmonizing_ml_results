from typing import Callable, List, Set, Tuple, TypeVar, Optional, Generic, Any, Union
import warnings
from allennlp.common.checks import ConfigurationError
from allennlp.data.tokenizers import Token

TypedSpan = Tuple[int, Tuple[int, int]]
TypedStringSpan = Tuple[str, Tuple[int, int]]

class InvalidTagSequence(Exception):
    def __init__(self, tag_sequence: Optional[List[str]] = None) -> None:
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self) -> str:
        return ' '.join(self.tag_sequence) if self.tag_sequence else ''

T = TypeVar('T', str, Token)

def enumerate_spans(
    sentence: List[T],
    offset: int = 0,
    max_span_width: Optional[int] = None,
    min_span_width: int = 1,
    filter_function: Optional[Callable[[List[T]], bool]] = None
) -> List[Tuple[int, int]]:
    max_span_width = max_span_width or len(sentence)
    filter_function = filter_function or (lambda x: True)
    spans: List[Tuple[int, int]] = []
    for start_index in range(len(sentence)):
        last_end_index = min(start_index + max_span_width, len(sentence))
        first_end_index = min(start_index + min_span_width - 1, len(sentence))
        for end_index in range(first_end_index, last_end_index):
            start = offset + start_index
            end = offset + end_index
            if filter_function(sentence[slice(start_index, end_index + 1)]):
                spans.append((start, end))
    return spans

def bio_tags_to_spans(
    tag_sequence: List[str],
    classes_to_ignore: Optional[List[str]] = None
) -> List[TypedStringSpan]:
    classes_to_ignore = classes_to_ignore or []
    spans: Set[TypedStringSpan] = set()
    span_start = 0
    span_end = 0
    active_conll_tag: Optional[str] = None
    for index, string_tag in enumerate(tag_sequence):
        bio_tag = string_tag[0]
        if bio_tag not in ['B', 'I', 'O']:
            raise InvalidTagSequence(tag_sequence)
        conll_tag = string_tag[2:]
        if bio_tag == 'O' or conll_tag in classes_to_ignore:
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            continue
        elif bio_tag == 'B':
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif bio_tag == 'I' and conll_tag == active_conll_tag:
            span_end += 1
        else:
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)

def iob1_tags_to_spans(
    tag_sequence: List[str],
    classes_to_ignore: Optional[List[str]] = None
) -> List[TypedStringSpan]:
    classes_to_ignore = classes_to_ignore or []
    spans: Set[TypedStringSpan] = set()
    span_start = 0
    span_end = 0
    active_conll_tag: Optional[str] = None
    prev_bio_tag: Optional[str] = None
    prev_conll_tag: Optional[str] = None
    for index, string_tag in enumerate(tag_sequence):
        curr_bio_tag = string_tag[0]
        curr_conll_tag = string_tag[2:]
        if curr_bio_tag not in ['B', 'I', 'O']:
            raise InvalidTagSequence(tag_sequence)
        if curr_bio_tag == 'O' or curr_conll_tag in classes_to_ignore:
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
        elif _iob1_start_of_chunk(prev_bio_tag, prev_conll_tag, curr_bio_tag, curr_conll_tag):
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = curr_conll_tag
            span_start = index
            span_end = index
        else:
            span_end += 1
        prev_bio_tag = string_tag[0]
        prev_conll_tag = string_tag[2:]
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)

def _iob1_start_of_chunk(
    prev_bio_tag: Optional[str],
    prev_conll_tag: Optional[str],
    curr_bio_tag: str,
    curr_conll_tag: str
) -> bool:
    if curr_bio_tag == 'B':
        return True
    if curr_bio_tag == 'I' and prev_bio_tag == 'O':
        return True
    if curr_bio_tag != 'O' and prev_conll_tag != curr_conll_tag:
        return True
    return False

def bioul_tags_to_spans(
    tag_sequence: List[str],
    classes_to_ignore: Optional[List[str]] = None
) -> List[TypedStringSpan]:
    spans: List[TypedStringSpan] = []
    classes_to_ignore = classes_to_ignore or []
    index = 0
    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label[0] == 'U':
            spans.append((label.partition('-')[2], (index, index)))
        elif label[0] == 'B':
            start = index
            while label[0] != 'L':
                index += 1
                if index >= len(tag_sequence):
                    raise InvalidTagSequence(tag_sequence)
                label = tag_sequence[index]
                if not (label[0] == 'I' or label[0] == 'L'):
                    raise InvalidTagSequence(tag_sequence)
            spans.append((label.partition('-')[2], (start, index)))
        elif label != 'O':
            raise InvalidTagSequence(tag_sequence)
        index += 1
    return [span for span in spans if span[0] not in classes_to_ignore]

def iob1_to_bioul(tag_sequence: List[str]) -> List[str]:
    warnings.warn("iob1_to_bioul has been replaced with 'to_bioul' to allow more encoding options.", FutureWarning)
    return to_bioul(tag_sequence)

def to_bioul(tag_sequence: List[str], encoding: str = 'IOB1') -> List[str]:
    if encoding not in {'IOB1', 'BIO'}:
        raise ConfigurationError(f"Invalid encoding {encoding} passed to 'to_bioul'.")

    def replace_label(full_label: str, new_label: str) -> str:
        parts = list(full_label.partition('-'))
        parts[0] = new_label
        return ''.join(parts)

    def pop_replace_append(
        in_stack: List[str],
        out_stack: List[str],
        new_label: str
    ) -> None:
        tag = in_stack.pop()
        new_tag = replace_label(tag, new_label)
        out_stack.append(new_tag)

    def process_stack(stack: List[str], out_stack: List[str]) -> None:
        if len(stack) == 1:
            pop_replace_append(stack, out_stack, 'U')
        else:
            recoded_stack: List[str] = []
            pop_replace_append(stack, recoded_stack, 'L')
            while len(stack) >= 2:
                pop_replace_append(stack, recoded_stack, 'I')
            pop_replace_append(stack, recoded_stack, 'B')
            recoded_stack.reverse()
            out_stack.extend(recoded_stack)

    bioul_sequence: List[str] = []
    stack: List[str] = []
    for label in tag_sequence:
        if label == 'O' and len(stack) == 0:
            bioul_sequence.append(label)
        elif label == 'O' and len(stack) > 0:
            process_stack(stack, bioul_sequence)
            bioul_sequence.append(label)
        elif label[0] == 'I':
            if len(stack) == 0:
                if encoding == 'BIO':
                    raise InvalidTagSequence(tag_sequence)
                stack.append(label)
            else:
                this_type = label.partition('-')[2]
                prev_type = stack[-1].partition('-')[2]
                if this_type == prev_type:
                    stack.append(label)
                else:
                    if encoding == 'BIO':
                        raise InvalidTagSequence(tag_sequence)
                    process_stack(stack, bioul_sequence)
                    stack.append(label)
        elif label[0] == 'B':
            if len(stack) > 0:
                process_stack(stack, bioul_sequence)
            stack.append(label)
        else:
            raise InvalidTagSequence(tag_sequence)
    if len(stack) > 0:
        process_stack(stack, bioul_sequence)
    return bioul_sequence

def bmes_tags_to_spans(
    tag_sequence: List[str],
    classes_to_ignore: Optional[List[str]] = None
) -> List[TypedStringSpan]:
    def extract_bmes_tag_label(text: str) -> Tuple[str, str]:
        bmes_tag = text[0]
        label = text[2:]
        return (bmes_tag, label)

    spans: List[Tuple[str, List[int]]] = []
    prev_bmes_tag: Optional[str] = None
    for index, tag in enumerate(tag_sequence):
        bmes_tag, label = extract_bmes_tag_label(tag)
        if bmes_tag in ('B', 'S'):
            spans.append((label, [index, index]))
        elif bmes_tag in ('M', 'E') and prev_bmes_tag in ('B', 'M') and (spans[-1][0] == label):
            spans[-1][1][1] = index
        else:
            spans.append((label, [index, index]))
        prev_bmes_tag = bmes_tag
    classes_to_ignore = classes_to_ignore or []
    return [(span[0], (span[1][0], span[1][1])) for span in spans if span[0] not in classes_to_ignore]
