            from prefect import flow, serve

            @flow
            def my_flow(name):
                print(f"hello {name}")

            @flow
            def my_other_flow(name):
                print(f"goodbye {name}")

            if __name__ == "__main__":
                hello_deploy = my_flow.to_deployment("hello", tags=["dev"])
                bye_deploy = my_other_flow.to_deployment("goodbye", tags=["dev"])
                serve(hello_deploy, bye_deploy)
            