from dataclasses import dataclass
from typing import Optional, ClassVar, List

@dataclass(init=False, repr=False)
class Token:
    __slots__: ClassVar[List[str]] = ['text', 'idx', 'idx_end', 'lemma_', 'pos_', 'tag_', 'dep_', 'ent_type_', 'text_id', 'type_id']

    text: Optional[str]
    idx: Optional[int]
    idx_end: Optional[int]
    lemma_: Optional[str]
    pos_: Optional[str]
    tag_: Optional[str]
    dep_: Optional[str]
    ent_type_: Optional[str]
    text_id: Optional[int]
    type_id: Optional[int]

    def __init__(
        self,
        text: Optional[str] = None,
        idx: Optional[int] = None,
        idx_end: Optional[int] = None,
        lemma_: Optional[str] = None,
        pos_: Optional[str] = None,
        tag_: Optional[str] = None,
        dep_: Optional[str] = None,
        ent_type_: Optional[str] = None,
        text_id: Optional[int] = None,
        type_id: Optional[int] = None
    ) -> None:
        assert text is None or isinstance(text, str)
        self.text = text
        self.idx = idx
        self.idx_end = idx_end
        self.lemma_ = lemma_
        self.pos_ = pos_
        self.tag_ = tag_
        self.dep_ = dep_
        self.ent_type_ = ent_type_
        self.text_id = text_id
        self.type_id = type_id

    def __str__(self) -> str:
        return self.text if self.text is not None else ''

    def __repr__(self) -> str:
        return self.__str__()

    def ensure_text(self) -> str:
        if self.text is None:
            raise ValueError('Unexpected null text for token')
        else:
            return self.text

def show_token(token: Token) -> str:
    return (
        f'{token.text} (idx: {token.idx}) (idx_end: {token.idx_end}) '
        f'(lemma: {token.lemma_}) (pos: {token.pos_}) (tag: {token.tag_}) '
        f'(dep: {token.dep_}) (ent_type: {token.ent_type_}) '
        f'(text_id: {token.text_id}) (type_id: {token.type_id}) '
    )