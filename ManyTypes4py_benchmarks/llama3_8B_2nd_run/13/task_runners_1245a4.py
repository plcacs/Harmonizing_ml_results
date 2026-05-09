        from prefect import flow
        from prefect_dask.task_runners import DaskTaskRunner

        @flow(task_runner=DaskTaskRunner)
        def my_flow():
            ...
        