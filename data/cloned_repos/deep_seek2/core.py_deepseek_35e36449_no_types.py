from typing import Any, Callable, Dict, Optional, Tuple, TypeVar
_C = TypeVar('_C')

def _validate_versions(datasets, load_versions, save_version):
    """Validates and synchronises dataset versions for loading and saving.

    Ensures consistency of dataset versions across a catalog, particularly
    for versioned datasets. It updates load versions and validates that all
    save versions are consistent.

    Args:
        datasets: A dictionary mapping dataset names to their instances.
            if None, no validation occurs.
        load_versions: A mapping between dataset names and versions
            to load.
        save_version: Version string to be used for ``save`` operations
            by all datasets with versioning enabled.

    Returns:
        Updated ``load_versions`` with load versions specified in the ``datasets``
            and resolved ``save_version``.

    Raises:
        VersionAlreadyExistsError: If a dataset's save version conflicts with
            the catalog's save version.
    """
    if not datasets:
        return (load_versions, save_version)
    cur_load_versions = load_versions.copy()
    cur_save_version = save_version
    for ds_name, ds in datasets.items():
        cur_ds = ds._dataset if ds.__class__.__name__ == 'CachedDataset' else ds
        if isinstance(cur_ds, AbstractVersionedDataset) and cur_ds._version:
            if cur_ds._version.load:
                cur_load_versions[ds_name] = cur_ds._version.load
            if cur_ds._version.save:
                cur_save_version = cur_save_version or cur_ds._version.save
                if cur_save_version != cur_ds._version.save:
                    raise VersionAlreadyExistsError(f'Cannot add a dataset `{ds_name}` with `{cur_ds._version.save}` save version. Save version set for the catalog is `{cur_save_version}`All datasets in the catalog must have the same save version.')
    return (cur_load_versions, cur_save_version)