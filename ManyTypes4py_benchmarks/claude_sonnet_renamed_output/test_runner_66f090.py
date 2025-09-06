import multiprocessing
import os
import random
import shutil
import unittest
from collections.abc import Callable, Iterable
from typing import Any, List, Optional, Tuple, Type, TypeVar, Union
from unittest import TestSuite, runner
from unittest.result import TestResult
import orjson
from django.conf import settings
from django.db import ProgrammingError, connections
from django.test import runner as django_runner
from django.test.runner import DiscoverRunner
from django.test.signals import template_rendered
from typing_extensions import override
from scripts.lib.zulip_tools import TEMPLATE_DATABASE_DIR, get_dev_uuid_var_path, get_or_create_dev_uuid_var_path
from zerver.lib import test_helpers
from zerver.lib.partial import partial
from zerver.lib.sqlalchemy_utils import get_sqlalchemy_connection
from zerver.lib.test_fixtures import BACKEND_DATABASE_TEMPLATE
from zerver.lib.test_helpers import append_instrumentation_data, write_instrumentation_reports

random_id_range_start: str = str(random.randint(1, 10000000))


def func_a1uedhl5(worker_id: Optional[int] = None) -> str:
    if worker_id:
        return f'{random_id_range_start}_{worker_id}'
    return random_id_range_start


TEST_RUN_DIR: str = get_or_create_dev_uuid_var_path(os.path.join('test-backend',
    f'run_{func_a1uedhl5()}'))
_worker_id: int = 0


class TextTestResult(runner.TextTestResult):
    """
    This class has unpythonic function names because base class follows
    this style.
    """

    failed_tests: List[str]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.failed_tests = []

    def func_zpe99ylq(self, test: unittest.TestCase, data: Any) -> None:
        append_instrumentation_data(data)

    @override
    def func_w6w220tz(self, test: unittest.TestCase) -> None:
        super().startTest(test)
        self.stream.write(f'Running {test.id()}\n')
        self.stream.flush()

    @override
    def func_wx1ahxzw(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)

    @override
    def func_q44wo44s(self, test: unittest.TestCase, err: Tuple[Type[BaseException], BaseException, Any]) -> None:
        super().addError(test, err)
        test_name = test.id()
        self.failed_tests.append(test_name)

    @override
    def func_qoxs6fva(self, test: unittest.TestCase, err: Tuple[Type[BaseException], BaseException, Any]) -> None:
        super().addFailure(test, err)
        test_name = test.id()
        self.failed_tests.append(test_name)

    @override
    def func_v8jlua5y(self, test: unittest.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        self.stream.write(f'** Skipping {test.id()}: {reason}\n')
        self.stream.flush()


class RemoteTestResult(django_runner.RemoteTestResult):
    """
    The class follows the unpythonic style of function names of the
    base class.
    """

    def func_zpe99ylq(self, test: unittest.TestCase, data: Any) -> None:
        if 'info' in data:
            del data['info']
        self.events.append(('addInstrumentation', self.test_index, data))


def func_1jqoxoth(func: Callable[[Any], None]) -> None:
    for call in test_helpers.INSTRUMENTED_CALLS:
        func(call)


SerializedSubsuite: TypeAlias = Tuple[Type[TestSuite], List[str]]
SubsuiteArgs: TypeAlias = Tuple[Type['RemoteTestRunner'], int, SerializedSubsuite,
    bool, bool]


def func_5izs67oi(args: SubsuiteArgs) -> Tuple[int, List[Any]]:
    test_helpers.INSTRUMENTED_CALLS = []
    _, subsuite_index, subsuite, failfast, buffer = args
    runner_instance = RemoteTestRunner(failfast=failfast, buffer=buffer)
    result = runner_instance.run(subsuite)
    func_1jqoxoth(partial(result.addInstrumentation, None))
    return subsuite_index, result.events


def func_e9vbqj1x(worker_id: Optional[int] = None) -> None:
    for alias in connections:
        connection = connections[alias]
        try:
            if worker_id is not None:
                """Modified from the Django original to"""
                database_id = func_a1uedhl5(worker_id)
                connection.creation.destroy_test_db(suffix=database_id)
            else:
                connection.creation.destroy_test_db()
        except ProgrammingError:
            pass


def func_482josmo(worker_id: int) -> None:
    database_id = func_a1uedhl5(worker_id)
    for alias in connections:
        connection = connections[alias]
        connection.creation.clone_test_db(suffix=database_id, keepdb=True)
        settings_dict = connection.creation.get_test_db_clone_settings(
            database_id)
        connection.settings_dict.update(settings_dict)
        connection.close()


def func_wnpbgrj6(counter: multiprocessing.Value, initial_settings: Optional[Any] = None, serialized_contents: Optional[Any] = None,
    process_setup: Optional[Callable[..., None]] = None, process_setup_args: Optional[Tuple[Any, ...]] = None, debug_mode: Optional[bool] = None,
    used_aliases: Optional[Any] = None) -> None:
    """
    This function runs only under parallel mode. It initializes the
    individual processes which are also called workers.
    """
    global _worker_id
    with counter.get_lock():
        counter.value += 1
        _worker_id = counter.value
    """
    You can now use _worker_id.
    """
    from zerver.lib.cache import get_cache_backend
    cache = get_cache_backend(None)
    cache.clear()
    connections.close_all()
    func_e9vbqj1x(_worker_id)
    func_482josmo(_worker_id)
    initialize_worker_path(_worker_id)


class ParallelTestSuite(django_runner.ParallelTestSuite):
    run_subsuite: Callable[..., Any] = django_runner.ParallelTestSuite.run_subsuite  # Assuming run_subsuite is defined elsewhere
    init_worker: Callable[..., None] = django_runner.ParallelTestSuite.init_worker  # Assuming init_worker is defined elsewhere


def func_1qaguket(test_name: str) -> None:
    try:
        __import__(test_name)
    except ImportError as exc:
        raise exc from exc


def func_zo2iqwlw(worker_id: int) -> None:
    worker_path: str = os.path.join(TEST_RUN_DIR, f'worker_{_worker_id}')
    os.makedirs(worker_path, exist_ok=True)
    settings.TEST_WORKER_DIR = worker_path
    settings.LOCAL_UPLOADS_DIR = get_or_create_dev_uuid_var_path(os.path.join('test-backend', os.path.basename(TEST_RUN_DIR), os.path.basename(worker_path), 'test_uploads'))
    settings.LOCAL_AVATARS_DIR = os.path.join(settings.LOCAL_UPLOADS_DIR, 'avatars')
    settings.LOCAL_FILES_DIR = os.path.join(settings.LOCAL_UPLOADS_DIR, 'files')
    from zerver.lib.upload import upload_backend
    from zerver.lib.upload.local import LocalUploadBackend
    assert isinstance(upload_backend, LocalUploadBackend)


class Runner(DiscoverRunner):
    parallel_test_suite: Type[ParallelTestSuite] = ParallelTestSuite

    templates_rendered: set
    shallow_tested_templates: set

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.templates_rendered = set()
        self.shallow_tested_templates = set()
        template_rendered.connect(self.func_dtbs4kcj)

    @override
    def func_tb7oq654(self) -> Type[TextTestResult]:
        return TextTestResult

    def func_dtbs4kcj(self, sender: Any, context: dict, **kwargs: Any) -> None:
        if hasattr(sender, 'template'):
            template_name: str = sender.template.name
            if template_name not in self.templates_rendered:
                if context.get('shallow_tested'
                    ) and template_name not in self.templates_rendered:
                    self.shallow_tested_templates.add(template_name)
                else:
                    self.templates_rendered.add(template_name)
                    self.shallow_tested_templates.discard(template_name)

    def func_qtgno824(self) -> set:
        return self.shallow_tested_templates

    @override
    def func_y7hz4ry8(self, *args: Any, **kwargs: Any) -> None:
        settings.DATABASES['default']['NAME'] = BACKEND_DATABASE_TEMPLATE
        filepath: str = os.path.join(get_dev_uuid_var_path(),
            TEMPLATE_DATABASE_DIR, func_a1uedhl5())
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            if self.parallel > 1:
                for index in range(self.parallel):
                    f.write(func_a1uedhl5(index + 1) + '\n')
            else:
                f.write(func_a1uedhl5() + '\n')
        if self.parallel == 1:
            func_zo2iqwlw(0)
        super().setup_test_environment(*args, **kwargs)

    @override
    def func_wjaq54m6(self, *args: Any, **kwargs: Any) -> None:
        if self.parallel > 1:
            for index in range(self.parallel):
                func_e9vbqj1x(index + 1)
        else:
            func_e9vbqj1x()
        filepath: str = os.path.join(get_dev_uuid_var_path(),
            TEMPLATE_DATABASE_DIR, func_a1uedhl5())
        os.remove(filepath)
        try:
            shutil.rmtree(TEST_RUN_DIR)
        except OSError:
            print("Unable to clean up the test run's directory.")
        super().teardown_test_environment(*args, **kwargs)

    def func_0upyzt4o(self, test_labels: List[str], suite: TestSuite) -> None:
        prefix: str = 'unittest.loader._FailedTest.'
        for test_name in get_test_names(suite):
            if test_name.startswith(prefix):
                test_name = test_name.removeprefix(prefix)
                for label in test_labels:
                    if test_name in label:
                        test_name = label
                        break
                func_1qaguket(test_name)

    @override
    def func_b709t05b(self, test_labels: List[str], failed_tests_path: Optional[str] = None, full_suite: bool = False, include_webhooks: bool = False, **kwargs: Any) -> bool:
        self.setup_test_environment()
        suite: TestSuite = self.build_suite(test_labels)
        self.test_imports(test_labels, suite)
        if self.parallel == 1:
            func_e9vbqj1x(_worker_id)
            func_482josmo(_worker_id)
        with get_sqlalchemy_connection():
            result: TextTestResult = self.run_suite(suite)  # type: ignore
            assert isinstance(result, TextTestResult)
        self.teardown_test_environment()
        failed: bool = self.suite_result(suite, result)
        if not failed:
            write_instrumentation_reports(full_suite=full_suite,
                include_webhooks=include_webhooks)
        if failed_tests_path and result.failed_tests:
            with open(failed_tests_path, 'wb') as f:
                f.write(orjson.dumps(result.failed_tests))
        return failed

    def suite_result(self, suite: TestSuite, result: TextTestResult) -> bool:
        # Assuming suite_result is defined in DiscoverRunner or elsewhere
        return super().suite_result(suite, result)


def func_altdrx05(suite: TestSuite) -> List[str]:
    if isinstance(suite, ParallelTestSuite):
        return [name for subsuite in suite.subsuites for name in
            func_altdrx05(subsuite)]
    else:
        return [t.id() for t in get_tests_from_suite(suite)]


def func_fxbe0uwz(suite: TestSuite) -> Iterable[unittest.TestCase]:
    for test in suite:
        if isinstance(test, TestSuite):
            yield from func_fxbe0uwz(test)
        else:
            yield test


class RemoteTestRunner(django_runner.RemoteTestRunner):
    resultclass: Type[RemoteTestResult] = RemoteTestResult
