def test_execute_query_as_report_executor(owner_names: List[str], creator_name: Optional[str], config: List[Union[FixedExecutor, ExecutorType]], expected_result: Union[str, Exception], mocker: MockerFixture, app_context: AppContext, get_user: Callable) -> None:

def test_execute_query_mutate_query_enabled(mocker: MockerFixture, app_context: AppContext, get_user: Callable) -> None:

def test_execute_query_mutate_query_disabled(mocker: MockerFixture, app_context: AppContext, get_user: Callable) -> None:

def test_execute_query_succeeded_no_retry(mocker: MockerFixture, app_context: AppContext) -> None:

def test_execute_query_succeeded_with_retries(mocker: MockerFixture, app_context: AppContext) -> None:

def test_execute_query_failed_no_retry(mocker: MockerFixture, app_context: AppContext) -> None:

def test_execute_query_failed_max_retries(mocker: MockerFixture, app_context: AppContext) -> None:

def test_get_alert_metadata_from_object(mocker: MockerFixture, app_context: AppContext, get_user: Callable) -> None:
