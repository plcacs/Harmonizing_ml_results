def get_inverse_hvp_lissa(vs: List[torch.Tensor], model: Model, used_params: Sequence[torch.Tensor], lissa_data_loader: DataLoader, damping: float, num_samples: int, scale: float) -> torch.Tensor:
    """
    This function approximates the product of the inverse of the Hessian and
    the vectors `vs` using LiSSA.

    Adapted from [github.com/kohpangwei/influence-release]
    (https://github.com/kohpangwei/influence-release/blob/0f656964867da6ddcca16c14b3e4f0eef38a7472/influence/genericNeuralNet.py#L475),
    the repo for [Koh, P.W., & Liang, P. (2017)](https://api.semanticscholar.org/CorpusID:13193974),
    and [github.com/xhan77/influence-function-analysis]
    (https://github.com/xhan77/influence-function-analysis/blob/78d5a967aba885f690d34e88d68da8678aee41f1/bert_util.py#L336),
    the repo for [Han, Xiaochuang et al. (2020)](https://api.semanticscholar.org/CorpusID:218628619).
    """
    inverse_hvps: List[torch.Tensor] = [torch.tensor(0) for _ in vs]
    for _ in Tqdm.tqdm(range(num_samples), desc='LiSSA samples', total=num_samples):
        cur_estimates = vs
        recursion_iter = Tqdm.tqdm(lissa_data_loader, desc='LiSSA depth', total=len(lissa_data_loader))
        for j, training_batch in enumerate(recursion_iter):
            model.zero_grad()
            train_output_dict = model(**training_batch)
            hvps: List[torch.Tensor] = get_hvp(train_output_dict['loss'], used_params, cur_estimates)
            cur_estimates = [v + (1 - damping) * cur_estimate - hvp / scale for v, cur_estimate, hvp in zip(vs, cur_estimates, hvps)]
            if j % 50 == 0 or j == len(lissa_data_loader) - 1:
                norm = np.linalg.norm(_flatten_tensors(cur_estimates).cpu().numpy())
                recursion_iter.set_description(desc=f'calculating inverse HVP, norm = {norm:.5f}')
        inverse_hvps = [inverse_hvp + cur_estimate / scale for inverse_hvp, cur_estimate in zip(inverse_hvps, cur_estimates)]
    return_ihvp = _flatten_tensors(inverse_hvps)
    return_ihvp /= num_samples
    return return_ihvp

def get_hvp(loss: torch.Tensor, params: Sequence[torch.Tensor], vectors: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """
    Get a Hessian-Vector Product (HVP) `Hv` for each Hessian `H` of the `loss`
    with respect to the one of the parameter tensors in `params` and the corresponding
    vector `v` in `vectors`.

    # Parameters

    loss : `torch.Tensor`
        The loss calculated from the output of the model.
    params : `Sequence[torch.Tensor]`
        Tunable and used parameters in the model that we will calculate the gradient and hessian
        with respect to.
    vectors : `Sequence[torch.Tensor]`
        The list of vectors for calculating the HVP.
    """
    assert len(params) == len(vectors)
    assert all((p.size() == v.size() for p, v in zip(params, vectors)))
    grads: List[torch.Tensor] = autograd.grad(loss, params, create_graph=True, retain_graph=True)
    hvp: List[torch.Tensor] = autograd.grad(grads, params, grad_outputs=vectors)
    return hvp

def _flatten_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Unwraps a list of parameters gradients

    # Returns

    `torch.Tensor`
        A tensor of shape `(x,)` where `x` is the total number of entires in the gradients.
    """
    views: List[torch.Tensor] = []
    for p in tensors:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)
