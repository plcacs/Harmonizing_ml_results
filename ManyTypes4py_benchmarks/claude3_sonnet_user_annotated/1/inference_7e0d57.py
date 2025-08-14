def _unshift_target_factors(sequence: np.ndarray, fill_last_with: int = C.EOS_ID) -> List[List[int]]:
    """
    Shifts back target factors so that they re-align with the words.

    :param sequence: Array of word ids. Shape: (bucketed_length, num_target_factors).
    """
    if len(sequence.shape) == 1 or sequence.shape[1] == 1:
        return sequence.tolist()
    num_factors_to_shift = sequence.shape[1] - 1
    _fillvalue: List[int] = num_factors_to_shift * [fill_last_with]
    _words: List[int] = sequence[:, 0].tolist()  # tokens from t==0 onwards
    _next_factors: List[List[int]] = sequence[1:, 1:].tolist()  # factors from t==1 onwards
    sequence = [(w, *fs) for w, fs in itertools.zip_longest(_words, _next_factors, fillvalue=_fillvalue)]  # type: ignore
    return sequence
