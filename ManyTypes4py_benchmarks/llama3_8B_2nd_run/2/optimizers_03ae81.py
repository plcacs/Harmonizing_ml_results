class Optimizer(torch.optim.Optimizer, Registrable):
    """This class just allows us to implement `Registrable` for Pytorch Optimizers."""
    default_implementation = 'adam'

    @staticmethod
    def default(model_parameters):
        return Optimizer.from_params(model_parameters=model_parameters, params=Params({}))

@Optimizer.register('multi')
class MultiOptimizer(Optimizer):
    """A `MultiOptimizer` creates a dictionary of `Optimizer`s keyed on some 'name'."""
    def __init__(self, model_parameters, optimizers: List[Dict[str, Any]], parameter_groups: Optional[List[Tuple[List[str], Dict[str, Any]]]] = None):
        ...

@Optimizer.register('adam')
class AdamOptimizer(Optimizer, torch.optim.Adam):
    """Registered as an `Optimizer` with name "adam"."""
    def __init__(self, model_parameters: List[Tuple[str, torch.Tensor]], parameter_groups: Optional[List[Tuple[List[str], Dict[str, Any]]]] = None, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.0, amsgrad: bool = False):
        ...

@Optimizer.register('sparse_adam')
class SparseAdamOptimizer(Optimizer, torch.optim.SparseAdam):
    """Registered as an `Optimizer` with name "sparse_adam"."""
    def __init__(self, model_parameters: List[Tuple[str, torch.Tensor]], parameter_groups: Optional[List[Tuple[List[str], Dict[str, Any]]]] = None, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08):
        ...

@Optimizer.register('adamw')
class AdamWOptimizer(Optimizer, torch.optim.AdamW):
    """Registered as an `Optimizer` with name "adamw"."""
    def __init__(self, model_parameters: List[Tuple[str, torch.Tensor]], parameter_groups: Optional[List[Tuple[List[str], Dict[str, Any]]]] = None, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.01, amsgrad: bool = False):
        ...

@Optimizer.register('huggingface_adamw')
class HuggingfaceAdamWOptimizer(Optimizer, transformers.AdamW):
    """Registered as an `Optimizer` with name "huggingface_adamw"."""
    def __init__(self, model_parameters: List[Tuple[str, torch.Tensor]], parameter_groups: Optional[List[Tuple[List[str], Dict[str, Any]]]] = None, lr: float = 1e-05, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.0, correct_bias: bool = True):
        ...

@Optimizer.register('huggingface_adafactor')
class HuggingfaceAdafactor(Optimizer, transformers.Adafactor):
    """Registered as an `Optimizer` with name "huggingface_adafactor"."""
    def __init__(self, model_parameters: List[Tuple[str, torch.Tensor]], parameter_groups: Optional[List[Tuple[List[str], Dict[str, Any]]]] = None, lr: Optional[float] = None, eps: Tuple[float, float] = (1e-30, 0.001), clip_threshold: float = 1.0, decay_rate: float = -0.8, beta1: Optional[float] = None, weight_decay: float = 0.0, scale_parameter: bool = True, relative_step: bool = True, warmup_init: bool = False):
        ...

@Optimizer.register('adagrad')
class AdagradOptimizer(Optimizer, torch.optim.Adagrad):
    """Registered as an `Optimizer` with name "adagrad"."""
    def __init__(self, model_parameters: List[Tuple[str, torch.Tensor]], parameter_groups: Optional[List[Tuple[List[str], Dict[str, Any]]]] = None, lr: float = 0.01, lr_decay: float = 0.0, weight_decay: float = 0.0, initial_accumulator_value: float = 0.0, eps: float = 1e-10):
        ...

@Optimizer.register('adadelta')
class AdadeltaOptimizer(Optimizer, torch.optim.Adadelta):
    """Registered as an `Optimizer` with name "adadelta"."""
    def __init__(self, model_parameters: List[Tuple[str, torch.Tensor]], parameter_groups: Optional[List[Tuple[List[str], Dict[str, Any]]]] = None, lr: float = 1.0, rho: float = 0.9, eps: float = 1e-06, weight_decay: float = 0.0):
        ...

@Optimizer.register('sgd')
class SgdOptimizer(Optimizer, torch.optim.SGD):
    """Registered as an `Optimizer` with name "sgd"."""
    def __init__(self, model_parameters: List[Tuple[str, torch.Tensor]], lr: float, parameter_groups: Optional[List[Tuple[List[str], Dict[str, Any]]]] = None, momentum: float = 0.0, dampening: float = 0, weight_decay: float = 0.0, nesterov: bool = False):
        ...

@Optimizer.register('rmsprop')
class RmsPropOptimizer(Optimizer, torch.optim.RMSprop):
    """Registered as an `Optimizer` with name "rmsprop"."""
    def __init__(self, model_parameters: List[Tuple[str, torch.Tensor]], parameter_groups: Optional[List[Tuple[List[str], Dict[str, Any]]]] = None, lr: float = 0.01, alpha: float = 0.99, eps: float = 1e-08, weight_decay: float = 0.0, momentum: float = 0.0, centered: bool = False):
        ...

@Optimizer.register('averaged_sgd')
class AveragedSgdOptimizer(Optimizer, torch.optim.ASGD):
    """Registered as an `Optimizer` with name "averaged_sgd"."""
    def __init__(self, model_parameters: List[Tuple[str, torch.Tensor]], parameter_groups: Optional[List[Tuple[List[str], Dict[str, Any]]]] = None, lr: float = 0.01, lambd: float = 0.0001, alpha: float = 0.75, t0: float = 1000000.0, weight_decay: float = 0.0):
        ...

@Optimizer.register('dense_sparse_adam')
class DenseSparseAdam(Optimizer, torch.optim.Optimizer):
    """NOTE: This class has been copied verbatim from the separate Dense and Sparse versions of Adam in Pytorch."""
    def __init__(self, model_parameters: List[Tuple[str, torch.Tensor]], parameter_groups: Optional[List[Tuple[List[str], Dict[str, Any]]]] = None, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08):
        ...
