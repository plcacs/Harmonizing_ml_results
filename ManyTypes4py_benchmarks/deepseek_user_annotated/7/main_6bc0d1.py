"""Tool for sorting imports alphabetically, and automatically separated into sections."""

import argparse
import functools
import json
import os
import sys
from gettext import gettext as _
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, Set, FrozenSet, Tuple, Iterable
from warnings import warn

from . import __version__, api, files, sections
from .exceptions import FileSkipped, ISortError, UnsupportedEncoding
from .format import create_terminal_printer
from .logo import ASCII_ART
from .profiles import profiles
from .settings import VALID_PY_TARGETS, Config, find_all_configs
from .utils import Trie
from .wrap_modes import WrapModes

DEPRECATED_SINGLE_DASH_ARGS: Set[str] = {
    "-ac",
    "-af",
    "-ca",
    "-cs",
    "-df",
    "-ds",
    "-dt",
    "-fas",
    "-fass",
    "-ff",
    "-fgw",
    "-fss",
    "-lai",
    "-lbt",
    "-le",
    "-ls",
    "-nis",
    "-nlb",
    "-ot",
    "-rr",
    "-sd",
    "-sg",
    "-sl",
    "-sp",
    "-tc",
    "-wl",
    "-ws",
}
QUICK_GUIDE: str = f"""
{ASCII_ART}

Nothing to do: no files or paths have been passed in!

Try one of the following:

    `isort .` - sort all Python files, starting from the current directory, recursively.
    `isort . --interactive` - Do the same, but ask before making any changes.
    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.
    `isort --help` - In-depth information about isort's available command-line options.

Visit https://pycqa.github.io/isort/ for complete information about how to use isort.
"""


class SortAttempt:
    def __init__(self, incorrectly_sorted: bool, skipped: bool, supported_encoding: bool) -> None:
        self.incorrectly_sorted: bool = incorrectly_sorted
        self.skipped: bool = skipped
        self.supported_encoding: bool = supported_encoding


def sort_imports(
    file_name: str,
    config: Config,
    check: bool = False,
    ask_to_apply: bool = False,
    write_to_stdout: bool = False,
    **kwargs: Any,
) -> Optional[SortAttempt]:
    incorrectly_sorted: bool = False
    skipped: bool = False
    try:
        if check:
            try:
                incorrectly_sorted = not api.check_file(file_name, config=config, **kwargs)
            except FileSkipped:
                skipped = True
            return SortAttempt(incorrectly_sorted, skipped, True)

        try:
            incorrectly_sorted = not api.sort_file(
                file_name,
                config=config,
                ask_to_apply=ask_to_apply,
                write_to_stdout=write_to_stdout,
                **kwargs,
            )
        except FileSkipped:
            skipped = True
        return SortAttempt(incorrectly_sorted, skipped, True)
    except (OSError, ValueError) as error:
        warn(f"Unable to parse file {file_name} due to {error}")
        return None
    except UnsupportedEncoding:
        if config.verbose:
            warn(f"Encoding not supported for {file_name}")
        return SortAttempt(incorrectly_sorted, skipped, False)
    except ISortError as error:
        _print_hard_fail(config, message=str(error))
        sys.exit(1)
    except Exception:
        _print_hard_fail(config, offending_file=file_name)
        raise


def _print_hard_fail(
    config: Config, offending_file: Optional[str] = None, message: Optional[str] = None
) -> None:
    """Fail on unrecoverable exception with custom message."""
    message = message or (
        f"Unrecoverable exception thrown when parsing {offending_file or ''}! "
        "This should NEVER happen.\n"
        "If encountered, please open an issue: https://github.com/PyCQA/isort/issues/new"
    )
    printer = create_terminal_printer(
        color=config.color_output, error=config.format_error, success=config.format_success
    )
    printer.error(message)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sort Python import definitions alphabetically "
        "within logical sections. Run with no arguments to see a quick "
        "start guide, otherwise, one or more files/directories/stdin must be provided. "
        "Use `-` as the first argument to represent stdin. Use --interactive to use the pre 5.0.0 "
        "interactive behavior."
        " "
        "If you've used isort 4 but are new to isort 5, see the upgrading guide: "
        "https://pycqa.github.io/isort/docs/upgrade_guides/5.0.0.html",
        add_help=False,  # prevent help option from appearing in "optional arguments" group
    )

    general_group = parser.add_argument_group("general options")
    target_group = parser.add_argument_group("target options")
    output_group = parser.add_argument_group("general output options")
    inline_args_group = output_group.add_mutually_exclusive_group()
    section_group = parser.add_argument_group("section output options")
    deprecated_group = parser.add_argument_group("deprecated options")

    general_group.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help=_("show this help message and exit"),
    )
    general_group.add_argument(
        "-V",
        "--version",
        action="store_true",
        dest="show_version",
        help="Displays the currently installed version of isort.",
    )
    general_group.add_argument(
        "--vn",
        "--version-number",
        action="version",
        version=__version__,
        help="Returns just the current version number without the logo",
    )
    general_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Shows verbose output, such as when files are skipped or when a check is successful.",
    )
    general_group.add_argument(
        "--only-modified",
        "--om",
        dest="only_modified",
        action="store_true",
        help="Suppresses verbose output for non-modified files.",
    )
    general_group.add_argument(
        "--dedup-headings",
        dest="dedup_headings",
        action="store_true",
        help="Tells isort to only show an identical custom import heading comment once, even if"
        " there are multiple sections with the comment set.",
    )
    general_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        dest="quiet",
        help="Shows extra quiet output, only errors are outputted.",
    )
    general_group.add_argument(
        "-d",
        "--stdout",
        help="Force resulting output to stdout, instead of in-place.",
        dest="write_to_stdout",
        action="store_true",
    )
    general_group.add_argument(
        "--overwrite-in-place",
        help="Tells isort to overwrite in place using the same file handle. "
        "Comes at a performance and memory usage penalty over its standard "
        "approach but ensures all file flags and modes stay unchanged.",
        dest="overwrite_in_place",
        action="store_true",
    )
    general_group.add_argument(
        "--show-config",
        dest="show_config",
        action="store_true",
        help="See isort's determined config, as well as sources of config options.",
    )
    general_group.add_argument(
        "--show-files",
        dest="show_files",
        action="store_true",
        help="See the files isort will be run against with the current config options.",
    )
    general_group.add_argument(
        "--df",
        "--diff",
        dest="show_diff",
        action="store_true",
        help="Prints a diff of all the changes isort would make to a file, instead of "
        "changing it in place",
    )
    general_group.add_argument(
        "-c",
        "--check-only",
        "--check",
        action="store_true",
        dest="check",
        help="Checks the file for unsorted / unformatted imports and prints them to the "
        "command line without modifying the file. Returns 0 when nothing would change and "
        "returns 1 when the file would be reformatted.",
    )
    general_group.add_argument(
        "--ws",
        "--ignore-whitespace",
        action="store_true",
        dest="ignore_whitespace",
        help="Tells isort to ignore whitespace differences when --check-only is being used.",
    )
    general_group.add_argument(
        "--sp",
        "--settings-path",
        "--settings-file",
        "--settings",
        dest="settings_path",
        help="Explicitly set the settings path or file instead of auto determining "
        "based on file location.",
    )
    general_group.add_argument(
        "--cr",
        "--config-root",
        dest="config_root",
        help="Explicitly set the config root for resolving all configs. When used "
        "with the --resolve-all-configs flag, isort will look at all sub-folders "
        "in this config root to resolve config files and sort files based on the "
        "closest available config(if any)",
    )
    general_group.add_argument(
        "--resolve-all-configs",
        dest="resolve_all_configs",
        action="store_true",
        help="Tells isort to resolve the configs for all sub-directories "
        "and sort files in terms of its closest config files.",
    )
    general_group.add_argument(
        "--profile",
        dest="profile",
        type=str,
        help="Base profile type to use for configuration. "
        f"Profiles include: {', '.join(profiles.keys())}. As well as any shared profiles.",
    )
    general_group.add_argument(
        "--old-finders",
        "--magic-placement",
        dest="old_finders",
        action="store_true",
        help="Use the old deprecated finder logic that relies on environment introspection magic.",
    )
    general_group.add_argument(
        "-j",
        "--jobs",
        help="Number of files to process in parallel. Negative value means use number of CPUs.",
        dest="jobs",
        type=int,
        nargs="?",
        const=-1,
    )
    general_group.add_argument(
        "--ac",
        "--atomic",
        dest="atomic",
        action="store_true",
        help="Ensures the output doesn't save if the resulting file contains syntax errors.",
    )
    general_group.add_argument(
        "--interactive",
        dest="ask_to_apply",
        action="store_true",
        help="Tells isort to apply changes interactively.",
    )
    general_group.add_argument(
        "--format-error",
        dest="format_error",
        help="Override the format used to print errors.",
    )
    general_group.add_argument(
        "--format-success",
        dest="format_success",
        help="Override the format used to print success.",
    )
    general_group.add_argument(
        "--srx",
        "--sort-reexports",
        dest="sort_reexports",
        action="store_true",
        help="Automatically sort all re-exports (module level __all__ collections)",
    )

    target_group.add_argument(
        "files", nargs="*", help="One or more Python source files that need their imports sorted."
    )
    target_group.add_argument(
        "--filter-files",
        dest="filter_files",
        action="store_true",
        help="Tells isort to filter files even when they are explicitly passed in as "
        "part of the CLI command.",
    )
    target_group.add_argument(
        "-s",
        "--skip",
        help="Files that isort should skip over. If you want to skip multiple "
        "files you should specify twice: --skip file1 --skip file2. Values can be "
        "file names, directory names or file paths. To skip all files in a nested path "
        "use --skip-glob.",
        dest="skip",
        action="append",
    )
    target_group.add_argument(
        "--extend-skip",
        help="Extends --skip to add additional files that isort should skip over. "
        "If you want to skip multiple "
        "files you should specify twice: --skip file1 --skip file2. Values can be "
        "file names, directory names or file paths. To skip all files in a nested path "
        "use --skip-glob.",
        dest="extend_skip",
        action="append",
    )
    target_group.add_argument(
        "--sg",
        "--skip-glob",
        help="Files that isort should skip over.",
        dest="skip_glob",
        action="append",
    )
    target_group.add_argument(
        "--extend-skip-glob",
        help="Additional files that isort should skip over (extending --skip-glob).",
        dest="extend_skip_glob",
        action="append",
    )
    target_group.add_argument(
        "--gitignore",
        "--skip-gitignore",
        action="store_true",
        dest="skip_gitignore",
        help="Treat project as a git repository and ignore files listed in .gitignore."
        "\nNOTE: This requires git to be installed and accessible from the same shell as isort.",
    )
    target_group.add_argument(
        "--ext",
        "--extension",
        "--supported-extension",
        dest="supported_extensions",
        action="append",
        help="Specifies what extensions isort can be run against.",
    )
    target_group.add_argument(
        "--blocked-extension",
        dest="blocked_extensions",
        action="append",
        help="Specifies what extensions isort can never be run against.",
    )
    target_group.add_argument(
        "--dont-follow-links",
        dest="dont_follow_links",
        action="store_true",
        help="Tells isort not to follow symlinks that are encountered when running recursively.",
    )
    target_group.add_argument(
        "--filename",
        dest="filename",
        help="Provide the filename associated with a stream.",
    )
    target_group.add_argument(
        "--allow-root",
        action="store_true",
        default=False,
        help="Tells isort not to treat / specially, allowing it to be run against the root dir.",
    )

    output_group.add_argument(
        "-a",
        "--add-import",
        dest="add_imports",
        action="append",
        help="Adds the specified import line to all files, "
        "automatically determining correct placement.",
    )
    output_group.add_argument(
        "--append",
        "--append-only",
        dest="append_only",
        action="store_true",
        help="Only adds the imports specified in --add-import if the file"
        " contains existing imports.",
    )
    output_group.add_argument(
        "--af",
        "--force-adds",
        dest="force_adds",
        action="store_true",
        help="Forces import adds even if the original file is empty.",
    )
    output_group.add_argument(
        "--rm",
        "--remove-import",
        dest="remove_imports",
        action="append",
        help="Removes the specified import from all files.",
    )
    output_group.add_argument(
        "--float-to-top",
        dest="float_to_top",
        action="store_true",
        help="Causes all non-indented imports to float to the top of the file having its imports "
        "sorted (immediately below the top of file comment).\n"
        "This can be an excellent shortcut for collecting imports every once in a while "
        "when you place them in the middle of a file to avoid context switching.\n\n"
        "*NOTE*: It currently doesn't work with cimports and introduces some extra over-head "
        "and a performance penalty.",
    )
    output_group.add_argument(
        "--dont-float-to-top",
        dest="dont_float_to_top",
        action="store_true",
        help="Forces --float-to-top setting off. See --float-to-top for more information.",
    )
    output_group.add_argument(
        "--ca",
        "--combine-as",
        dest="combine_as_imports",
        action="store_true",
        help="Combines as imports on the same line.",
    )
    output_group.add_argument(
        "--cs",
        "--combine-star",
        dest="combine_star",
        action="store_true",
        help="Ensures that if a star import is present, "
        "nothing else is imported from that namespace.",
    )
    output_group.add_argument(
        "-e",
        "--balanced",
        dest="balanced_wrapping",
        action="store_true",
        help="Balances wrapping to produce the most consistent line length possible",
    )
    output_group.add_argument(
        "--ff",
        "--from-first",
        dest="from_first",
        action="store_true",
        help="Switches the typical ordering preference, "
        "showing from imports first then straight ones.",
    )
    output_group.add_argument(
        "--fgw",
        "--force-grid-wrap",
        nargs="?",
        const=2,
        type=int,
        dest="force_grid_wrap",
        help="Force number of from imports (defaults to 2 when passed as CLI flag without value) "
        "to be grid wrapped regardless of line "
        "length. If 0 is passed in (the global default) only line length is considered.",
    )
    output_group.add_argument(
        "-i",
        "--indent",
        help='String to place for indents defaults to "    " (4 spaces).',
        dest="indent",
        type=str,
    )
    output_group.add_argument(
        "--lbi", "--lines-before-imports", dest="lines_before_imports", type=int
    )
    output_group.add_argument(
        "--lai", "--lines-after-imports", dest="lines_after_imports", type=int
    )
    output_group.add_argument(
        "--lbt", "--lines-between-types", dest="lines_between_types", type=int
    )
    output_group.add_argument(
        "--le",
        "--line-ending",
        dest="line_ending",
        help="Forces line endings to the specified value. "
        "If not set, values will be guessed per-file.",
    )
    output_group.add_argument(
        "--ls",
        "--length-sort",
        help="Sort imports by their string length.",
