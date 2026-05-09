"""
Prefect deployment steps for building and pushing Docker images.

These steps can be used in a `prefect.yaml` file to define the default
build steps for a group of deployments, or they can be used to define
the build step for a specific deployment.

!!! example
    Build a Docker image before deploying a flow:
    