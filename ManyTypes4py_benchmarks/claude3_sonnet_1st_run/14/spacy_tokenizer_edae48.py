from typing import List, Optional, Dict, Any, Iterator
import spacy
from spacy.tokens import Doc, Token as SpacyToken
from allennlp.common.util import get_spacy_model
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

@Tokenizer.register('spacy')
class SpacyTokenizer(Tokenizer):
    """
    A `Tokenizer` that uses spaCy's tokenizer.  It's fast and reasonable - this is the
    recommended `Tokenizer`. By default it will return allennlp Tokens,
    which are small, efficient NamedTuples (and are serializable). If you want
    to keep the original spacy tokens, pass keep_spacy_tokens=True.  Note that we leave one particular piece of
    post-processing for later: the decision of whether or not to lowercase the token.  This is for
    two reasons: (1) if you want to make two different casing decisions for whatever reason, you
    won't have to run the tokenizer twice, and more importantly (2) if you want to lowercase words
    for your word embedding, but retain capitalization in a character-level representation, we need
    to retain the capitalization here.

    Registered as a `Tokenizer` with name "spacy", which is currently the default.

    # Parameters

    language : `str`, optional, (default=`"en_core_web_sm"`)
        Spacy model name.
    pos_tags : `bool`, optional, (default=`False`)
        If `True`, performs POS tagging with spacy model on the tokens.
        Generally used in conjunction with :class:`~allennlp.data.token_indexers.pos_tag_indexer.PosTagIndexer`.
    parse : `bool`, optional, (default=`False`)
        If `True`, performs dependency parsing with spacy model on the tokens.
        Generally used in conjunction with :class:`~allennlp.data.token_indexers.pos_tag_indexer.DepLabelIndexer`.
    ner : `bool`, optional, (default=`False`)
        If `True`, performs dependency parsing with spacy model on the tokens.
        Generally used in conjunction with :class:`~allennlp.data.token_indexers.ner_tag_indexer.NerTagIndexer`.
    keep_spacy_tokens : `bool`, optional, (default=`False`)
        If `True`, will preserve spacy token objects, We copy spacy tokens into our own class by default instead
        because spacy Cython Tokens can't be pickled.
    split_on_spaces : `bool`, optional, (default=`False`)
        If `True`, will split by spaces without performing tokenization.
        Used when your data is already tokenized, but you want to perform pos, ner or parsing on the tokens.
    start_tokens : `Optional[List[str]]`, optional, (default=`None`)
        If given, these tokens will be added to the beginning of every string we tokenize.
    end_tokens : `Optional[List[str]]`, optional, (default=`None`)
        If given, these tokens will be added to the end of every string we tokenize.
    """

    def __init__(self, 
                 language: str = 'en_core_web_sm', 
                 pos_tags: bool = True, 
                 parse: bool = False, 
                 ner: bool = False, 
                 keep_spacy_tokens: bool = False, 
                 split_on_spaces: bool = False, 
                 start_tokens: Optional[List[str]] = None, 
                 end_tokens: Optional[List[str]] = None) -> None:
        self._language = language
        self._pos_tags = pos_tags
        self._parse = parse
        self._ner = ner
        self._split_on_spaces = split_on_spaces
        self.spacy = get_spacy_model(self._language, self._pos_tags, self._parse, self._ner)
        if self._split_on_spaces:
            self.spacy.tokenizer = _WhitespaceSpacyTokenizer(self.spacy.vocab)
        self._keep_spacy_tokens = keep_spacy_tokens
        self._start_tokens = start_tokens or []
        self._start_tokens.reverse()
        self._is_version_3 = spacy.__version__ >= '3.0'
        self._end_tokens = end_tokens or []

    def _sanitize(self, tokens: List[SpacyToken]) -> List[Token]:
        """
        Converts spaCy tokens to allennlp tokens. Is a no-op if
        keep_spacy_tokens is True
        """
        if not self._keep_spacy_tokens:
            tokens = [Token(token.text, token.idx, token.idx + len(token.text), token.lemma_, token.pos_, token.tag_, token.dep_, token.ent_type_) for token in tokens]
        for start_token in self._start_tokens:
            tokens.insert(0, Token(start_token, 0))
        for end_token in self._end_tokens:
            tokens.append(Token(end_token, -1))
        return tokens

    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        if self._is_version_3:
            return [self._sanitize(_remove_spaces(tokens)) for tokens in self.spacy.pipe(texts, n_process=-1)]
        else:
            return [self._sanitize(_remove_spaces(tokens)) for tokens in self.spacy.pipe(texts, n_threads=-1)]

    def tokenize(self, text: str) -> List[Token]:
        return self._sanitize(_remove_spaces(self.spacy(text)))

    def _to_params(self) -> Dict[str, Any]:
        return {'type': 'spacy', 'language': self._language, 'pos_tags': self._pos_tags, 'parse': self._parse, 'ner': self._ner, 'keep_spacy_tokens': self._keep_spacy_tokens, 'split_on_spaces': self._split_on_spaces, 'start_tokens': self._start_tokens, 'end_tokens': self._end_tokens}

class _WhitespaceSpacyTokenizer:
    """
    Spacy doesn't assume that text is tokenised. Sometimes this
    is annoying, like when you have gold data which is pre-tokenised,
    but Spacy's tokenisation doesn't match the gold. This can be used
    as follows:
    nlp = spacy.load("en_core_web_md")
    # hack to replace tokenizer with a whitespace tokenizer
    nlp.tokenizer = _WhitespaceSpacyTokenizer(nlp.vocab)
    ... use nlp("here is some text") as normal.
    """

    def __init__(self, vocab: spacy.vocab.Vocab) -> None:
        self.vocab = vocab

    def __call__(self, text: str) -> Doc:
        words = text.split(' ')
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

def _remove_spaces(tokens: Doc) -> List[SpacyToken]:
    return [token for token in tokens if not token.is_space]
