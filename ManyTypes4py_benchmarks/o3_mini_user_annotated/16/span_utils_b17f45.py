from typing import Callable, List, Set, Tuple, TypeVar, Optional

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
        return " ".join(self.tag_sequence) if self.tag_sequence is not None else ""

T = TypeVar("T", str, Token)

def enumerate_spans(
    sentence: List[T],
    offset: int = 0,
    max_span_width: Optional[int] = None,
    min_span_width: int = 1,
    filter_function: Optional[Callable[[List[T]], bool]] = None,
) -> List[Tuple[int, int]]:
    """
    Given a sentence, return all token spans within the sentence. Spans are `inclusive`.
    Additionally, you can provide a maximum and minimum span width, which will be used
    to exclude spans outside of this range.

    Finally, you can provide a function mapping `List[T] -> bool`, which will
    be applied to every span to decide whether that span should be included. This
    allows filtering by length, regex matches, pos tags or any Spacy `Token`
    attributes, for example.
    """
    max_span_width = max_span_width or len(sentence)
    filter_function = filter_function or (lambda x: True)
    spans: List[Tuple[int, int]] = []

    for start_index in range(len(sentence)):
        last_end_index = min(start_index + max_span_width, len(sentence))
        first_end_index = min(start_index + min_span_width - 1, len(sentence))
        for end_index in range(first_end_index, last_end_index):
            start = offset + start_index
            end = offset + end_index
            # add 1 to end index because span indices are inclusive.
            if filter_function(sentence[slice(start_index, end_index + 1)]):
                spans.append((start, end))
    return spans

def bio_tags_to_spans(
    tag_sequence: List[str],
    classes_to_ignore: Optional[List[str]] = None,
) -> List[TypedStringSpan]:
    """
    Given a sequence corresponding to BIO tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
    as otherwise it is possible to get a perfect precision score whilst still predicting
    ill-formed spans in addition to the correct spans. This function works properly when
    the spans are unlabeled (i.e., your labels are simply "B", "I", and "O").
    """
    classes_to_ignore = classes_to_ignore or []
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start: int = 0
    span_end: int = 0
    active_conll_tag: Optional[str] = None
    for index, string_tag in enumerate(tag_sequence):
        # Actual BIO tag.
        bio_tag = string_tag[0]
        if bio_tag not in ["B", "I", "O"]:
            raise InvalidTagSequence(tag_sequence)
        conll_tag = string_tag[2:]
        if bio_tag == "O" or conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
        elif bio_tag == "B":
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif bio_tag == "I" and conll_tag == active_conll_tag:
            # We're inside a span.
            span_end += 1
        else:
            # This is the case the bio label is an "I", but either:
            # 1) the span hasn't started - i.e. an ill formed span.
            # 2) The span is an I tag for a different conll annotation.
            # We'll process the previous span if it exists, but also
            # include this span. This is important, because otherwise,
            # a model may get a perfect F1 score whilst still including
            # false positive ill-formed spans.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
    # Last token might have been a part of a valid span.
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)

def iob1_tags_to_spans(
    tag_sequence: List[str],
    classes_to_ignore: Optional[List[str]] = None,
) -> List[TypedStringSpan]:
    """
    Given a sequence corresponding to IOB1 tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e., those where "B-LABEL" is not preceded
    by "I-LABEL" or "B-LABEL").

    # Parameters

    tag_sequence : `List[str]`, required.
        The integer class labels for a sequence.
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.

    # Returns

    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    classes_to_ignore = classes_to_ignore or []
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start: int = 0
    span_end: int = 0
    active_conll_tag: Optional[str] = None
    prev_bio_tag: Optional[str] = None
    prev_conll_tag: Optional[str] = None
    for index, string_tag in enumerate(tag_sequence):
        curr_bio_tag: str = string_tag[0]
        curr_conll_tag: str = string_tag[2:]

        if curr_bio_tag not in ["B", "I", "O"]:
            raise InvalidTagSequence(tag_sequence)
        if curr_bio_tag == "O" or curr_conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
        elif _iob1_start_of_chunk(prev_bio_tag, prev_conll_tag, curr_bio_tag, curr_conll_tag):
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = curr_conll_tag
            span_start = index
            span_end = index
        else:
            # bio_tag == "I" and curr_conll_tag == active_conll_tag
            # We're continuing a span.
            span_end += 1

        prev_bio_tag = string_tag[0]
        prev_conll_tag = string_tag[2:]
    # Last token might have been a part of a valid span.
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)

def _iob1_start_of_chunk(
    prev_bio_tag: Optional[str],
    prev_conll_tag: Optional[str],
    curr_bio_tag: str,
    curr_conll_tag: str,
) -> bool:
    if curr_bio_tag == "B":
        return True
    if curr_bio_tag == "I" and prev_bio_tag == "O":
        return True
    if curr_bio_tag != "O" and prev_conll_tag != curr_conll_tag:
        return True
    return False

def bioul_tags_to_spans(
    tag_sequence: List[str],
    classes_to_ignore: Optional[List[str]] = None,
) -> List[TypedStringSpan]:
    """
    Given a sequence corresponding to BIOUL tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are not allowed and will raise `InvalidTagSequence`.
    This function works properly when the spans are unlabeled (i.e., your labels are
    simply "B", "I", "O", "U", and "L").
    """
    spans: List[TypedStringSpan] = []
    classes_to_ignore = classes_to_ignore or []
    index: int = 0
    while index < len(tag_sequence):
        label: str = tag_sequence[index]
        if label[0] == "U":
            spans.append((label.partition("-")[2], (index, index)))
        elif label[0] == "B":
            start: int = index
            while label[0] != "L":
                index += 1
                if index >= len(tag_sequence):
                    raise InvalidTagSequence(tag_sequence)
                label = tag_sequence[index]
                if not (label[0] == "I" or label[0] == "L"):
                    raise InvalidTagSequence(tag_sequence)
            spans.append((label.partition("-")[2], (start, index)))
        else:
            if label != "O":
                raise InvalidTagSequence(tag_sequence)
        index += 1
    return [span for span in spans if span[0] not in classes_to_ignore]

def iob1_to_bioul(tag_sequence: List[str]) -> List[str]:
    warnings.warn(
        "iob1_to_bioul has been replaced with 'to_bioul' to allow more encoding options.",
        FutureWarning,
    )
    return to_bioul(tag_sequence)

def to_bioul(tag_sequence: List[str], encoding: str = "IOB1") -> List[str]:
    """
    Given a tag sequence encoded with IOB1 labels, recode to BIOUL.

    In the IOB1 scheme, I is a token inside a span, O is a token outside
    a span and B is the beginning of span immediately following another
    span of the same type.

    In the BIO scheme, I is a token inside a span, O is a token outside
    a span and B is the beginning of a span.
    """
    if encoding not in {"IOB1", "BIO"}:
        raise ConfigurationError(f"Invalid encoding {encoding} passed to 'to_bioul'.")

    def replace_label(full_label: str, new_label: str) -> str:
        parts = list(full_label.partition("-"))
        parts[0] = new_label
        return "".join(parts)

    def pop_replace_append(in_stack: List[str], out_stack: List[str], new_label: str) -> None:
        tag = in_stack.pop()
        new_tag = replace_label(tag, new_label)
        out_stack.append(new_tag)

    def process_stack(stack: List[str], out_stack: List[str]) -> None:
        if len(stack) == 1:
            pop_replace_append(stack, out_stack, "U")
        else:
            recoded_stack: List[str] = []
            pop_replace_append(stack, recoded_stack, "L")
            while len(stack) >= 2:
                pop_replace_append(stack, recoded_stack, "I")
            pop_replace_append(stack, recoded_stack, "B")
            recoded_stack.reverse()
            out_stack.extend(recoded_stack)

    bioul_sequence: List[str] = []
    stack: List[str] = []

    for label in tag_sequence:
        if label == "O" and len(stack) == 0:
            bioul_sequence.append(label)
        elif label == "O" and len(stack) > 0:
            process_stack(stack, bioul_sequence)
            bioul_sequence.append(label)
        elif label[0] == "I":
            if len(stack) == 0:
                if encoding == "BIO":
                    raise InvalidTagSequence(tag_sequence)
                stack.append(label)
            else:
                this_type: str = label.partition("-")[2]
                prev_type: str = stack[-1].partition("-")[2]
                if this_type == prev_type:
                    stack.append(label)
                else:
                    if encoding == "BIO":
                        raise InvalidTagSequence(tag_sequence)
                    process_stack(stack, bioul_sequence)
                    stack.append(label)
        elif label[0] == "B":
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
    classes_to_ignore: Optional[List[str]] = None,
) -> List[TypedStringSpan]:
    """
    Given a sequence corresponding to BMES tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
    as otherwise it is possible to get a perfect precision score whilst still predicting
    ill-formed spans in addition to the correct spans.
    This function works properly when the spans are unlabeled (i.e., your labels are
    simply "B", "M", "E" and "S").
    """
    def extract_bmes_tag_label(text: str) -> Tuple[str, str]:
        bmes_tag: str = text[0]
        label: str = text[2:]
        return bmes_tag, label

    spans: List[Tuple[str, List[int]]] = []
    prev_bmes_tag: Optional[str] = None
    for index, tag in enumerate(tag_sequence):
        bmes_tag, label = extract_bmes_tag_label(tag)
        if bmes_tag in ("B", "S"):
            spans.append((label, [index, index]))
        elif bmes_tag in ("M", "E") and prev_bmes_tag in ("B", "M") and spans[-1][0] == label:
            spans[-1][1][1] = index
        else:
            spans.append((label, [index, index]))
        prev_bmes_tag = bmes_tag

    classes_to_ignore = classes_to_ignore or []
    return [
        (span[0], (span[1][0], span[1][1]))
        for span in spans
        if span[0] not in classes_to_ignore
    ]