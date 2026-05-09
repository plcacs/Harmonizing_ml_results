class Optimizer(torch.optim.Optimizer, Registrable):
    """
    This class just allows us to implement `Registrable` for Pytorch Optimizers.  We do something a
    little bit different with `Optimizers`, because they are implemented as classes in PyTorch, and
    we want to use those classes.  To make things easy, we just inherit from those classes, using
    multiple inheritance to also inherit from `Optimizer`.  The only reason we do this is to make
    type inference on parameters possible, so we can construct these objects using our configuration
    framework. If you are writing your own script, you can safely ignore these classes and just use
    the `torch.optim` classes directly.

    If you are implementing one of these classes, the `model_parameters` and `parameter_groups`
    arguments to `__init__` are important, and should always be present.  The trainer will pass
    the trainable parameters in the model to the optimizer using the name `model_parameters`, so if
    you use a different name, your code will crash.  Nothing will technically crash if you use a
    name other than `parameter_groups` for your second argument, it will just be annoyingly
    inconsistent.

    Most subclasses of `Optimizer` take both a `model_parameters` and a `parameter_groups`
    constructor argument. The `model_parameters` argument does not get an entry in a typical
    AllenNlp configuration file, but the `parameter_groups` argument does (if you want a non-default
    value).  See the documentation for the `make_parameter_groups` function for more information on
    how the `parameter_groups` argument should be specified.
    """
    default_implementation = 'adam'

    @staticmethod
    def default(model_parameters: List[Tuple[str, torch.nn.Parameter]], params: Params = Params({})) -> 'Optimizer':
        return Optimizer.from_params(model_parameters=model_parameters, params=params)

@Optimizer.register('multi')
class MultiOptimizer(Optimizer):
    """
    A `MultiOptimizer` creates a dictionary of `Optimizer`s keyed on some 'name'.
    Each Optimizer contains its own set of parameters which are obtained using
    regex matches for certain model parameters.

    This optimizer works by taking in a parameter `optimizers` which contains a list of `Optimizers`
    with their keyword arguments, and a parameter `parameter_groups`, which contains regexes and their
    corresponding optimizer and optional non-default optimizer options for this group.
    The regexes in the parameter groups are assigned to their optimizer based on the 'name' argument
    where the 'name' value should be the same for the optimizer and parameter group.
    You should specify a default optimizer with 'name': 'default' which will be used for all
    parameters which didn't obtain a regex match or when your parameter group doesn't contain a 'name'
    parameter.

    # Parameters

    optimizers: `List[Dict[str, Any]]`
        A list of optimizers to use. Each entry in the list is a dictionary of keyword arguments. A 'name'
        keyword argument should be given which will serve as the key to match the optimizer with a
        specific parameter group. You should also supply an entry for the default parameter group,
        e.g. 'name': 'default'.

    parameter_groups:  `List[Tuple[List[str], Dict[str, Any]]`, optional (default = `None`)
        See the docstring of `make_parameter_groups` for what this parameter should look like. It
        should follow the same format as there, except an additional 'optimizer_name' argument should be
        provided to match this group to its own optimizer. Optimizer options can also be set for this
        group which will override the default options.
    """

    def __init__(self, model_parameters: List[Tuple[str, torch.nn.Parameter]], 
                 optimizers: Dict[str, Dict[str, Any]], 
                 parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None):
        if 'default' not in optimizers:
            raise ConfigurationError("No optimizer was provided for the 'default' group. Please provide an Optimizer under the name 'default'")
        optimizer_name_to_parameter_groups = {optimizer_name: [] for optimizer_name in optimizers.keys()}
        for parameter_group in parameter_groups:
            regexes, pg_overrides = parameter_group
            optimizer_name = pg_overrides.get('optimizer_name', 'default')
            optimizer_name_to_parameter_groups[optimizer_name].append(parameter_group)
        optimizer_name_to_model_parameters = {optimizer_name: [] for optimizer_name in optimizers.keys()}
        for model_parameter_tuple in model_parameters:
            parameter_name, parameter_tensor = model_parameter_tuple
            for regexes, pg_overrides in parameter_groups:
                if any((re.search(regex, parameter_name) for regex in regexes)):
                    optimizer_name = pg_overrides.get('optimizer_name', 'default')
                    optimizer_name_to_model_parameters[optimizer_name].append(model_parameter_tuple)
                    break
            else:
                optimizer_name_to_model_parameters['default'].append(model_parameter_tuple)
        for optimizer_name, optimizer_parameters in optimizer_name_to_model_parameters.items():
            if optimizer_name != 'default' and len(optimizer_parameters) == 0:
                raise ConfigurationError(f"Optimizer '{optimizer_name}' did not receive any parameters. If you are using `parameter_groups`, please make sure that the regexes you have provided match the desired model parameters, or that the `name` value of this optimizer  matches that of the parameter group you are trying to assign to it. Alternatively, you can remove this optimizer from the provided `optimizers` if it is not relevant to a particular parameter group.")
        if len(optimizer_name_to_model_parameters['default']) == 0:
            del optimizers['default']
            del optimizer_name_to_model_parameters['default']
            del optimizer_name_to_parameter_groups['default']
        self.optimizers = {optimizer_name: lazy_optimizer.construct(model_parameters=optimizer_name_to_model_parameters[optimizer_name], parameter_groups=optimizer_name_to_parameter_groups[optimizer_name]) for optimizer_name, lazy_optimizer in optimizers.items()}
        parameter_groups = copy.deepcopy(parameter_groups)
        for parameter_group in parameter_groups:
            regexes, pg_overrides = parameter_group
            optimizer_name = pg_overrides.get('optimizer_name', 'default')
            optimizer = self.optimizers[optimizer_name]
            for key, value in optimizer.defaults.items():
                if key not in pg_overrides:
                    pg_overrides[key] = value
        made_parameter_groups = make_parameter_groups(model_parameters, parameter_groups)
        if 'default' in self.optimizers:
            for key, value in self.optimizers['default'].defaults.items():
                made_parameter_groups[-1][key] = value
        super().__init__(made_parameter_groups, {})

    def step(self) -> None:
        """
        Takes an optimization step for each optimizer.
        """
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self) -> Dict[str, Any]:
        """
        Creates an object `optimizer_state_dict`, which is a dictionary mapping an optimizer key to its
        `state_dict`. This dictionary is used as the value for 'optimizer' in the 'training_states' dictionary in
        the `gradient_descent` `Trainer`, e.g.
        