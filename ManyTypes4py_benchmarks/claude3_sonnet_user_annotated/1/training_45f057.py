def cleanup_params_files(output_folder: str, max_to_keep: int, checkpoint: int, best_checkpoint: int, keep_first: bool,
                         max_params_files_to_cache: int, cache_metric: str, cache_strategy: str) -> None:
    """
    Deletes oldest parameter files from a model folder.

    :param output_folder: Folder where param files are located.
    :param max_to_keep: Maximum number of files to keep, negative to keep all.
    :param checkpoint: Current checkpoint (i.e. index of last params file created).
    :param best_checkpoint: Best checkpoint. The parameter file corresponding to this checkpoint will not be deleted.
    :param keep_first: Don't delete the first checkpoint.
    :param max_params_files_to_cache: Maximum number of best param files to cache.
    :param cache_metric: Metric to determine best param files.
    :param cache_strategy: Strategy to select 'best' param files.
    """
    if max_to_keep <= 0:
        return

    # make sure we keep N best params files from .metrics file according to strategy.
    top_n: Set[int] = set()
    metrics_path = os.path.join(output_folder, C.METRICS_NAME)

    if max_params_files_to_cache > 0 and os.path.exists(metrics_path):
        maximize = C.METRIC_MAXIMIZE[cache_metric]
        points = utils.get_validation_metric_points(model_path=output_folder, metric=cache_metric)

        if cache_strategy == C.AVERAGE_BEST:
            # N best scoring points
            top = average.strategy_best(points, max_params_files_to_cache, maximize)

        elif cache_strategy == C.AVERAGE_LAST:
            # N sequential points ending with overall best
            top = average.strategy_last(points, max_params_files_to_cache, maximize)

        elif cache_strategy == C.AVERAGE_LIFESPAN:
            # Track lifespan of every "new best" point
            # Points dominated by a previous better point have lifespan 0
            top = average.strategy_lifespan(points, max_params_files_to_cache, maximize)
        else:
            raise RuntimeError("Unknown strategy, options are: %s" % C.AVERAGE_CHOICES)

        top_n = set([x[1] for x in top])

    # get rid of params files that are neither among the latest, nor among the best
    existing_files = glob.glob(os.path.join(output_folder, C.PARAMS_PREFIX + "*"))
    params_name_with_dir = os.path.join(output_folder, C.PARAMS_NAME)

    for n in range(1 if keep_first else 0, max(1, checkpoint - max_to_keep + 1)):
        if n != best_checkpoint:
            param_fname_n = params_name_with_dir % n
            if param_fname_n in existing_files and n not in top_n:
                try:
                    if os.path.isdir(param_fname_n):
                        # DeepSpeed mode initially saves checkpoint directories
                        shutil.rmtree(param_fname_n)
                    else:
                        os.remove(param_fname_n)
                except FileNotFoundError:
                    # This can be occur on file systems with higher latency,
                    # such as distributed file systems.  While repeated
                    # occurrences of this warning may indicate a problem, seeing
                    # one or two warnings during training is usually fine.
                    logger.warning('File has already been removed: %s', param_fname_n)
