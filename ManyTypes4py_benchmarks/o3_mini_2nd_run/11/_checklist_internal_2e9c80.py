#!/usr/bin/env python3
"""
The `checklist` subcommand allows you to conduct behavioural
testing for your model's predictions using a trained model and its
[`Predictor`](../predictors/predictor.md#predictor) wrapper.
"""
from typing import Optional, Dict, Any, List, IO
import argparse
import sys
import json
import logging
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

logger: logging.Logger = logging.getLogger(__name__)

try:
    from allennlp.confidence_checks.task_checklists.task_suite import TaskSuite
except ImportError:
    raise

@Subcommand.register('checklist')
class CheckList(Subcommand):
    def add_subparser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        description: str = 'Run the specified model through a checklist suite.'
        subparser: argparse.ArgumentParser = parser.add_parser(
            self.name,
            description=description,
            help='Run a trained model through a checklist suite.'
        )
        subparser.add_argument('archive_file', type=str, help='The archived model to make predictions with')
        subparser.add_argument('task', type=str, help='The name of the task suite')
        subparser.add_argument('--checklist-suite', type=str, help='The checklist suite path')
        subparser.add_argument('--capabilities', nargs='+', default=[], help='An optional list of strings of capabilities. Eg. "[Vocabulary, Robustness]"')
        subparser.add_argument('--max-examples', type=int, default=None, help='Maximum number of examples to check per test.')
        subparser.add_argument('--task-suite-args', type=str, default='', help='An optional JSON structure used to provide additional parameters to the task suite')
        subparser.add_argument('--print-summary-args', type=str, default='', help='An optional JSON structure used to provide additional parameters for printing test summary')
        subparser.add_argument('--output-file', type=str, help='Path to output file')
        subparser.add_argument('--cuda-device', type=int, default=-1, help='ID of GPU to use (if any)')
        subparser.add_argument('--predictor', type=str, help='Optionally specify a specific predictor to use')
        subparser.add_argument('--predictor-args', type=str, default='', help='An optional JSON structure used to provide additional parameters to the predictor')
        subparser.set_defaults(func=_run_suite)
        return subparser

def _get_predictor(args: argparse.Namespace) -> Predictor:
    check_for_gpu(args.cuda_device)
    archive = load_archive(args.archive_file, cuda_device=args.cuda_device)
    predictor_args_str: str = args.predictor_args.strip()
    if len(predictor_args_str) <= 0:
        predictor_args: Dict[str, Any] = {}
    else:
        predictor_args = json.loads(predictor_args_str)
    return Predictor.from_archive(archive, args.predictor, extra_args=predictor_args)

def _get_task_suite(args: argparse.Namespace) -> "TaskSuite":
    available_tasks: List[str] = TaskSuite.list_available()
    if args.task in available_tasks:
        suite_name: str = args.task
    else:
        raise ConfigurationError(f"'{args.task}' is not a recognized task suite. Available tasks are: {available_tasks}.")
    file_path: Optional[str] = args.checklist_suite
    task_suite_args_str: str = args.task_suite_args.strip()
    if len(task_suite_args_str) <= 0:
        task_suite_args: Dict[str, Any] = {}
    else:
        task_suite_args = json.loads(task_suite_args_str)
    return TaskSuite.constructor(name=suite_name, suite_file=file_path, extra_args=task_suite_args)

class _CheckListManager:
    def __init__(self,
                 task_suite: "TaskSuite",
                 predictor: Predictor,
                 capabilities: Optional[List[str]] = None,
                 max_examples: Optional[int] = None,
                 output_file: Optional[str] = None,
                 print_summary_args: Optional[Dict[str, Any]] = None) -> None:
        self._task_suite: "TaskSuite" = task_suite
        self._predictor: Predictor = predictor
        self._capabilities: Optional[List[str]] = capabilities
        self._max_examples: Optional[int] = max_examples
        self._output_file: Optional[IO[str]] = None if output_file is None else open(output_file, 'w')
        self._print_summary_args: Dict[str, Any] = print_summary_args or {}
        if capabilities:
            self._print_summary_args['capabilities'] = capabilities

    def run(self) -> None:
        self._task_suite.run(self._predictor, capabilities=self._capabilities, max_examples=self._max_examples)
        output_file: IO[str] = self._output_file or sys.stdout
        self._task_suite.summary(file=output_file, **self._print_summary_args)
        if self._output_file is not None:
            self._output_file.close()

def _run_suite(args: argparse.Namespace) -> None:
    task_suite: "TaskSuite" = _get_task_suite(args)
    predictor: Predictor = _get_predictor(args)
    print_summary_args_str: str = args.print_summary_args.strip()
    if len(print_summary_args_str) <= 0:
        print_summary_args: Dict[str, Any] = {}
    else:
        print_summary_args = json.loads(print_summary_args_str)
    capabilities: List[str] = args.capabilities
    max_examples: Optional[int] = args.max_examples
    manager: _CheckListManager = _CheckListManager(task_suite, predictor, capabilities, max_examples, args.output_file, print_summary_args)
    manager.run()