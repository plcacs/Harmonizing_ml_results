def _cache_wrapper(self, class_func: callable) -> callable:
    @lru_cache(maxsize=self.forward_pass_cache_size)
    def cache_func(*args: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, List[pt.Tensor], Optional[pt.Tensor]]:
        return class_func(*args)

    return cache_func
