@sync_compatible
async def deploy(
    *deployments: RunnerDeployment,
    work_pool_name: Optional[str] = None,
    image: Optional[DockerImage] = None,
    build: bool = True,
    push: bool = True,
    print_next_steps_message: bool = True,
    ignore_warnings: bool = False,
) -> List[UUID]:
    """
    Deploy the provided list of deployments to dynamic infrastructure via a
    work pool.

    By default, calling this function will build a Docker image for the deployments, push it to a
    registry, and create each deployment via the Prefect API that will run the corresponding
    flow on the given schedule.

    If you want to use an existing image, you can pass `build=False` to skip building and pushing
    an image.

    Args:
        *deployments: A list of deployments to deploy.
        work_pool_name: The name of the work pool to use for these deployments. Defaults to
            the value of `PREFECT_DEFAULT_WORK_POOL_NAME`.
        image: The name of the Docker image to build, including the registry and
            repository. Pass a DockerImage instance to customize the Dockerfile used
            and build arguments.
        build: Whether or not to build a new image for the flow. If False, the provided
            image will be used as-is and pulled at runtime.
        push: Whether or not to skip pushing the built image to a registry.
        print_next_steps_message: Whether or not to print a message with next steps
            after deploying the deployments.

    Returns:
        A list of deployment IDs for the created/updated deployments.

    Examples:
        Deploy a group of flows to a work pool:

        