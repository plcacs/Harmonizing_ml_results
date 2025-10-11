#!/usr/bin/env python3
"""
Generic LLM analysis script for mypy results, any-ratio, and type info data.
This script compares performance across original, partial, and full typing scenarios for any LLM.
"""

import json
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class AnalysisResult:
    """Container for analysis results."""

    llm_name: str
    total_files: int
    non_any_ratios: Tuple[float, float, float]  # O, U, H (compiled files only)
    non_any_ratios_all: Tuple[float, float, float]  # O, U, H (all files)
    precision_ratios: Tuple[float, float, float]  # O, U, H
    compiled_files: List[str]


class LLMAnalyzer:
    def __init__(self, llm_name: str):
        self.llm_name = llm_name.lower()
        self.base_path = "ManyTypes4py_benchmarks"
        self.setup_file_paths()

    def setup_file_paths(self):
        """Setup file paths based on LLM name."""
        if self.llm_name == "claude":
            # Claude-specific file paths based on actual directory structure
            self.mypy_files = {
                "partial": "mypy_results/mypy_outputs/partial_typed/mypy_results_claude3_sonnet_partially_typed_files_with_errors.json",
                "full": "mypy_results/mypy_outputs/mypy_results_claude3_sonnet_user_annotated_with_errors.json",
                "original": "mypy_results/mypy_outputs/mypy_results_claude3_sonnet_1st_run_with_errors.json",
            }

            self.any_ratio_files = {
                "partial": "Type_info_collector/Section_04/per_file_any_percentage_partially_typed_user_annotated/Claude3-Sonnet_Partially_Typed/per_file_any_percentage.json",
                "full": "Type_info_collector/Section_04/per_file_any_percentage_partially_typed_user_annotated/Claude3-Sonnet_User_Annotated/per_file_any_percentage.json",
                "original": "Type_info_collector/Section_04/per_file_any_percentage/Claude3-Sonnet_1st_run/per_file_any_percentage.json",
            }

            self.typeinfo_files = {
                "partial": "Type_info_collector/Type_info_LLMS/Type_info_claude3_sonnet_partially_typed_files_benchmarks.json",
                "full": "Type_info_collector/Type_info_LLMS/Type_info_claude3_sonnet_user_annotated_benchmarks.json",
                "original": "Type_info_collector/Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",
            }
        elif self.llm_name == "o3-mini":
            self.mypy_files = {
                "partial": "mypy_results/mypy_outputs/partial_typed/mypy_results_o3_mini_partially_typed_files_with_errors.json",
                "full": "mypy_results/mypy_outputs/mypy_results_o3_mini_user_annotated_with_errors.json",
                "original": "mypy_results/mypy_outputs/mypy_results_o3_mini_1st_run_with_errors.json",
            }

            self.any_ratio_files = {
                "partial": "Type_info_collector/Section_04/per_file_any_percentage_partially_typed_user_annotated/O3-mini_Partially_Typed/per_file_any_percentage.json",
                "full": "Type_info_collector/Section_04/per_file_any_percentage_partially_typed_user_annotated/O3-mini_User_Annotated/per_file_any_percentage.json",
                "original": "Type_info_collector/Section_04/per_file_any_percentage/O3-mini_1st_run/per_file_any_percentage.json",
            }

            self.typeinfo_files = {
                "partial": "Type_info_collector/Type_info_LLMS/Type_info_o3_mini_partially_typed_files_benchmarks.json",
                "full": "Type_info_collector/Type_info_LLMS/Type_info_o3_mini_user_annotated_benchmarks.json",
                "original": "Type_info_collector/Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
            }
        elif self.llm_name == "deepseek":
            self.mypy_files = {
                "partial": "mypy_results/mypy_outputs/partial_typed/mypy_results_deepseek_partially_typed_files_with_errors.json",
                "full": "mypy_results/mypy_outputs/mypy_results_deepseek_user_annotated_with_errors.json",
                "original": "mypy_results/mypy_outputs/mypy_results_deepseek_with_errors.json",
            }

            self.any_ratio_files = {
                "partial": "Type_info_collector/Section_04/per_file_any_percentage_partially_typed_user_annotated/DeepSeek_Partially_Typed/per_file_any_percentage.json",
                "full": "Type_info_collector/Section_04/per_file_any_percentage_partially_typed_user_annotated/DeepSeek_User_Annotated/per_file_any_percentage.json",
                "original": "Type_info_collector/Section_04/per_file_any_percentage/DeepSeek_1st_run/per_file_any_percentage.json",
            }

            self.typeinfo_files = {
                "partial": "Type_info_collector/Type_info_LLMS/Type_info_deepseek_partially_typed_files_benchmarks.json",
                "full": "Type_info_collector/Type_info_LLMS/Type_info_deepseek_user_annotated_benchmarks.json",
                "original": "Type_info_collector/Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
            }

        else:
            raise ValueError(
                f"LLM '{self.llm_name}' not supported. Currently only 'claude' is supported."
            )

    def load_json_file(self, file_path: str) -> Dict:
        """Load and parse a JSON file."""
        full_path = os.path.join(self.base_path, file_path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: File not found: {full_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file {full_path}: {e}")
            return {}

    def find_common_compiled_files(self) -> List[str]:
        """Find files that have isCompiled: true in all three mypy result files."""
        print(f"Loading mypy results for {self.llm_name.upper()}...")

        mypy_data = {}
        for scenario, file_path in self.mypy_files.items():
            mypy_data[scenario] = self.load_json_file(file_path)
            print(f"  {scenario}: {len(mypy_data[scenario])} files")

        # Find files that compile successfully in all scenarios
        all_files = (
            set(mypy_data["original"].keys())
            & set(mypy_data["partial"].keys())
            & set(mypy_data["full"].keys())
        )

        compiled_files = []
        for filename in all_files:
            if (
                mypy_data["original"][filename].get("isCompiled", False)
                and mypy_data["partial"][filename].get("isCompiled", False)
                and mypy_data["full"][filename].get("isCompiled", False)
            ):
                compiled_files.append(filename)

        print(f"Found {len(compiled_files)} files that compile in all scenarios")
        print(f"Total common files: {len(all_files)}")

        return compiled_files

    def calculate_non_any_ratios(
        self, compiled_files: List[str]
    ) -> Tuple[float, float, float]:
        """Calculate ratios for which scenario has the highest non-Any ratio (filtered by compiled files)."""
        print("Loading any-ratio data...")

        any_ratio_data = {}
        for scenario, file_path in self.any_ratio_files.items():
            any_ratio_data[scenario] = self.load_json_file(file_path)
            print(f"  {scenario}: {len(any_ratio_data[scenario])} files")

        # Initialize counters
        counters = {"O": 0, "U": 0, "H": 0}
        scenario_stats = defaultdict(list)

        print(
            f"Analyzing {len(compiled_files)} files for non-Any ratios (compiled files only)..."
        )

        tie_stats = {"no_tie": 0, "two_way_tie": 0, "three_way_tie": 0}

        for filename in compiled_files:
            # Get any percentages for each scenario
            original_any = (
                any_ratio_data["original"]
                .get(filename, {})
                .get("any_percentage", 100.0)
            )
            partial_any = (
                any_ratio_data["partial"]
                .get(filename, {})
                .get("any_percentage", original_any)
            )
            full_any = (
                any_ratio_data["full"]
                .get(filename, {})
                .get("any_percentage", original_any)
            )

            # Calculate non-any ratios (100 - any_percentage)
            ratios = {
                "U": 100 - original_any,
                "H": 100 - partial_any,
                "O": 100 - full_any,
            }

            scenario_stats["original"].append(ratios["U"])
            scenario_stats["partial"].append(ratios["H"])
            scenario_stats["full"].append(ratios["O"])

            # Find the highest non-any ratio
            max_ratio = max(ratios.values())

            # Count how many scenarios tie for the highest ratio
            winners = [
                scenario for scenario, ratio in ratios.items() if ratio == max_ratio
            ]
            if len(winners) == 1:
                tie_stats["no_tie"] += 1
                # Only count when there's a clear winner (no ties)
                counters[winners[0]] += 1
            elif len(winners) == 2:
                tie_stats["two_way_tie"] += 1
                # Don't increment counters for ties
            else:
                tie_stats["three_way_tie"] += 1
                # Don't increment counters for ties

        # Calculate ratios
        total_files = len(compiled_files)
        ratios = (
            counters["O"] / total_files,
            counters["U"] / total_files,
            counters["H"] / total_files,
        )

        # Print detailed statistics
        print(f"Non-Any Ratio Statistics (compiled files only):")
        for scenario, values in scenario_stats.items():
            avg_ratio = sum(values) / len(values)
            print(f"  {scenario}: avg non-any ratio = {avg_ratio:.2f}%")

        print(f"Tie Analysis:")
        print(f"  No ties: {tie_stats['no_tie']} files")
        print(f"  Two-way ties: {tie_stats['two_way_tie']} files")
        print(f"  Three-way ties: {tie_stats['three_way_tie']} files")

        # Store file counts for display
        self.non_any_file_counts = {
            "O": counters["O"],
            "U": counters["U"],
            "H": counters["H"],
        }

        return ratios

    def calculate_non_any_ratios_all_files(self) -> Tuple[float, float, float]:
        """Calculate ratios for which scenario has the highest non-Any ratio (all files, ignoring compilation status)."""
        print("Loading any-ratio data for all files...")

        any_ratio_data = {}
        for scenario, file_path in self.any_ratio_files.items():
            any_ratio_data[scenario] = self.load_json_file(file_path)
            print(f"  {scenario}: {len(any_ratio_data[scenario])} files")

        # Find all files that exist in at least one scenario
        all_files = set()
        for scenario_data in any_ratio_data.values():
            all_files.update(scenario_data.keys())

        print(f"Found {len(all_files)} total files across all scenarios")

        # Initialize counters
        counters = {"O": 0, "U": 0, "H": 0}
        scenario_stats = defaultdict(list)

        print(f"Analyzing {len(all_files)} files for non-Any ratios (all files)...")

        for filename in all_files:
            # Get any percentages for each scenario
            original_any = (
                any_ratio_data["original"]
                .get(filename, {})
                .get("any_percentage", 100.0)
            )
            partial_any = (
                any_ratio_data["partial"]
                .get(filename, {})
                .get("any_percentage", original_any)
            )
            full_any = (
                any_ratio_data["full"]
                .get(filename, {})
                .get("any_percentage", original_any)
            )

            # Calculate non-any ratios (100 - any_percentage)
            ratios = {
                "U": 100 - original_any,
                "H": 100 - partial_any,
                "O": 100 - full_any,
            }

            scenario_stats["original"].append(ratios["U"])
            scenario_stats["partial"].append(ratios["H"])
            scenario_stats["full"].append(ratios["O"])

            # Find the highest non-any ratio
            max_ratio = max(ratios.values())

            # Count which scenarios have the highest ratio (only when there's a clear winner)
            winners = [
                scenario for scenario, ratio in ratios.items() if ratio == max_ratio
            ]
            if len(winners) == 1:
                counters[winners[0]] += 1

        # Calculate ratios
        total_files = len(all_files)
        ratios = (
            counters["O"] / total_files,
            counters["U"] / total_files,
            counters["H"] / total_files,
        )

        # Print detailed statistics
        print(f"Non-Any Ratio Statistics (all files):")
        for scenario, values in scenario_stats.items():
            avg_ratio = sum(values) / len(values)
            print(f"  {scenario}: avg non-any ratio = {avg_ratio:.2f}%")

        # Store file counts for display
        self.non_any_file_counts_all = {
            "O": counters["O"],
            "U": counters["U"],
            "H": counters["H"],
        }

        return ratios

    def calculate_precision_ratios(
        self, compiled_files: List[str]
    ) -> Tuple[float, float, float]:
        """Calculate ratios for which scenario is more precise using typeinfo data."""
        print("Loading typeinfo data...")

        typeinfo_data = {}
        for scenario, file_path in self.typeinfo_files.items():
            typeinfo_data[scenario] = self.load_json_file(file_path)
            print(f"  {scenario}: {len(typeinfo_data[scenario])} files")

        # Initialize counters
        counters = {"O": 0, "U": 0, "H": 0}
        precision_stats = defaultdict(list)

        print(f"Analyzing {len(compiled_files)} files for precision...")

        for filename in compiled_files:
            original_info = typeinfo_data["original"].get(filename, {})
            partial_info = typeinfo_data["partial"].get(filename, original_info)
            full_info = typeinfo_data["full"].get(filename, original_info)

            # Calculate precision metrics
            precisions = {
                "U": self.calculate_file_precision(original_info),
                "H": self.calculate_file_precision(partial_info),
                "O": self.calculate_file_precision(full_info),
            }

            precision_stats["original"].append(precisions["U"])
            precision_stats["partial"].append(precisions["H"])
            precision_stats["full"].append(precisions["O"])

            # Find the highest precision
            max_precision = max(precisions.values())

            # Count which scenarios have the highest precision (only when there's a clear winner)
            winners = [
                scenario
                for scenario, precision in precisions.items()
                if precision == max_precision
            ]
            if len(winners) == 1:
                counters[winners[0]] += 1

        # Calculate ratios
        total_files = len(compiled_files)
        ratios = (
            counters["O"] / total_files,
            counters["U"] / total_files,
            counters["H"] / total_files,
        )

        # Print detailed statistics
        print(f"Precision Statistics:")
        for scenario, values in precision_stats.items():
            avg_precision = sum(values) / len(values)
            print(f"  {scenario}: avg precision = {avg_precision:.2f}%")

        # Store file counts for display
        self.precision_file_counts = {
            "O": counters["O"],
            "U": counters["U"],
            "H": counters["H"],
        }

        return ratios

    def calculate_file_precision(self, file_info: Dict) -> float:
        """Calculate precision score for a file based on typeinfo data."""
        if not file_info:
            return 0.0

        total_types = 0
        specific_types = 0

        for function_name, type_list in file_info.items():
            for type_info in type_list:
                if type_info.get("category") in ["arg", "return"]:
                    types = type_info.get("type", [])
                    for type_val in types:
                        total_types += 1
                        # Count as specific if it's not empty and not 'Any'
                        if type_val and type_val.strip() and type_val != "Any":
                            specific_types += 1

        if total_types == 0:
            return 0.0

        return (specific_types / total_types) * 100

    def get_type_precision_score(self, type_val: str) -> int:
        """Assign precision score to a type annotation (0-100 scale)."""
        if not type_val or not type_val.strip():
            return 0  # No annotation
        
        type_val = type_val.strip()
        
        if type_val.lower() == "any":
            return 10  # Any type
        
        # Generic types (medium precision)
        generic_types = ["list", "dict", "tuple", "set", "frozenset"]
        if any(generic in type_val.lower() for generic in generic_types):
            return 50
        
        # Optional types (medium-high precision)
        if "optional" in type_val.lower() or "union" in type_val.lower():
            return 70
        
        # Built-in types (high precision)
        builtin_types = ["str", "int", "float", "bool", "bytes"]
        if any(builtin in type_val.lower() for builtin in builtin_types):
            return 80
        
        # Specific generic types like List[str], Dict[str, int] (very high precision)
        if "[" in type_val and "]" in type_val:
            return 90
        
        # Custom types, classes, etc. (high precision)
        return 85

    def analyze_organic_gains_losses(self, compiled_files: List[str]) -> Dict:
        """Analyze organic gains and losses across scenarios."""
        print("Loading typeinfo data for organic analysis...")
        
        typeinfo_data = {}
        for scenario, file_path in self.typeinfo_files.items():
            typeinfo_data[scenario] = self.load_json_file(file_path)
            print(f"  {scenario}: {len(typeinfo_data[scenario])} files")
        
        results = {
            'H_vs_U': {'gains': 0, 'losses': 0, 'neutral': 0, 'total': 0},
            'O_vs_H': {'gains': 0, 'losses': 0, 'neutral': 0, 'total': 0},
            'O_vs_U': {'gains': 0, 'losses': 0, 'neutral': 0, 'total': 0}
        }
        
        print(f"Analyzing organic gains/losses for {len(compiled_files)} files...")
        
        for filename in compiled_files:
            original_info = typeinfo_data['original'].get(filename, {})
            partial_info = typeinfo_data['partial'].get(filename, original_info)
            full_info = typeinfo_data['full'].get(filename, original_info)
            
            # Extract all parameters from all scenarios
            all_params = set()
            for scenario_info in [original_info, partial_info, full_info]:
                for func_name, func_data in scenario_info.items():
                    for param_info in func_data:
                        if param_info.get('category') in ['arg', 'return']:
                            param_key = f"{func_name}::{param_info.get('name', 'return')}::{param_info.get('category')}"
                            all_params.add(param_key)
            
            # Analyze each parameter
            for param_key in all_params:
                func_name, param_name, category = param_key.split('::')
                
                # Get type scores for each scenario
                u_score = self.get_type_precision_score(self._get_param_type(original_info, func_name, param_name, category))
                h_score = self.get_type_precision_score(self._get_param_type(partial_info, func_name, param_name, category))
                o_score = self.get_type_precision_score(self._get_param_type(full_info, func_name, param_name, category))
                
                # Analyze H vs U (Partial vs Original)
                if h_score > u_score:
                    results['H_vs_U']['gains'] += 1
                elif h_score < u_score:
                    results['H_vs_U']['losses'] += 1
                else:
                    results['H_vs_U']['neutral'] += 1
                results['H_vs_U']['total'] += 1
                
                # Analyze O vs H (Full vs Partial)
                if o_score > h_score:
                    results['O_vs_H']['gains'] += 1
                elif o_score < h_score:
                    results['O_vs_H']['losses'] += 1
                else:
                    results['O_vs_H']['neutral'] += 1
                results['O_vs_H']['total'] += 1
                
                # Analyze O vs U (Full vs Original)
                if o_score > u_score:
                    results['O_vs_U']['gains'] += 1
                elif o_score < u_score:
                    results['O_vs_U']['losses'] += 1
                else:
                    results['O_vs_U']['neutral'] += 1
                results['O_vs_U']['total'] += 1
        
        return results

    def _get_param_type(self, file_info: Dict, func_name: str, param_name: str, category: str) -> str:
        """Extract type annotation for a specific parameter."""
        if func_name not in file_info:
            return ""
        
        for param_info in file_info[func_name]:
            if (param_info.get('category') == category and 
                param_info.get('name', 'return') == param_name):
                types = param_info.get('type', [])
                return types[0] if types else ""
        
        return ""

    def calculate_pairwise_non_any_comparisons(self, compiled_files: List[str]) -> Dict:
        """Calculate detailed pairwise comparisons for non-Any ratios."""
        print("Loading any-ratio data for pairwise comparisons...")
        
        any_ratio_data = {}
        for scenario, file_path in self.any_ratio_files.items():
            any_ratio_data[scenario] = self.load_json_file(file_path)
        
        results = {
            'H_U': {'wins': [], 'losses': [], 'ties': 0},
            'O_U': {'wins': [], 'losses': [], 'ties': 0},
            'O_H': {'wins': [], 'losses': [], 'ties': 0}
        }
        
        print(f"Analyzing pairwise non-Any ratios for {len(compiled_files)} files...")
        
        for filename in compiled_files:
            # Get any percentages for each scenario
            original_any = any_ratio_data["original"].get(filename, {}).get("any_percentage", 100.0)
            partial_any = any_ratio_data["partial"].get(filename, {}).get("any_percentage", original_any)
            full_any = any_ratio_data["full"].get(filename, {}).get("any_percentage", original_any)
            
            # Calculate non-any ratios (100 - any_percentage)
            ratios = {
                "U": 100 - original_any,  # Untyped (original)
                "H": 100 - partial_any,   # Half-typed (partial)
                "O": 100 - full_any,      # Original types (full)
            }
            
            # H vs U comparison
            if ratios["H"] > ratios["U"]:
                results['H_U']['wins'].append(ratios["H"])
            elif ratios["H"] < ratios["U"]:
                results['H_U']['losses'].append(ratios["H"])
            else:
                results['H_U']['ties'] += 1
            
            # O vs U comparison
            if ratios["O"] > ratios["U"]:
                results['O_U']['wins'].append(ratios["O"])
            elif ratios["O"] < ratios["U"]:
                results['O_U']['losses'].append(ratios["O"])
            else:
                results['O_U']['ties'] += 1
            
            # O vs H comparison
            if ratios["O"] > ratios["H"]:
                results['O_H']['wins'].append(ratios["O"])
            elif ratios["O"] < ratios["H"]:
                results['O_H']['losses'].append(ratios["O"])
            else:
                results['O_H']['ties'] += 1
        
        # Calculate summary statistics
        total_files = len(compiled_files)
        summary = {}
        
        for comparison, data in results.items():
            win_count = len(data['wins'])
            loss_count = len(data['losses'])
            win_pct = (win_count / total_files) * 100
            loss_pct = (loss_count / total_files) * 100
            win_avg = sum(data['wins']) / len(data['wins']) if data['wins'] else 0
            loss_avg = sum(data['losses']) / len(data['losses']) if data['losses'] else 0
            
            summary[comparison] = {
                'win_count': win_count,
                'loss_count': loss_count,
                'win_pct': win_pct,
                'loss_pct': loss_pct,
                'win_avg': win_avg,
                'loss_avg': loss_avg,
                'ties': data['ties']
            }
        
        return summary

    def calculate_pairwise_precision_comparisons(self, compiled_files: List[str]) -> Dict:
        """Calculate pairwise precision comparisons."""
        print("Loading typeinfo data for pairwise precision comparisons...")
        
        typeinfo_data = {}
        for scenario, file_path in self.typeinfo_files.items():
            typeinfo_data[scenario] = self.load_json_file(file_path)
        
        results = {
            'H_U': {'wins': 0, 'losses': 0, 'ties': 0},
            'O_U': {'wins': 0, 'losses': 0, 'ties': 0},
            'O_H': {'wins': 0, 'losses': 0, 'ties': 0}
        }
        
        print(f"Analyzing pairwise precision for {len(compiled_files)} files...")
        
        for filename in compiled_files:
            original_info = typeinfo_data['original'].get(filename, {})
            partial_info = typeinfo_data['partial'].get(filename, original_info)
            full_info = typeinfo_data['full'].get(filename, original_info)
            
            # Calculate precision for each scenario
            precisions = {
                "U": self.calculate_file_precision(original_info),  # Untyped (original)
                "H": self.calculate_file_precision(partial_info),   # Half-typed (partial)
                "O": self.calculate_file_precision(full_info),      # Original types (full)
            }
            
            # H vs U comparison
            if precisions["H"] > precisions["U"]:
                results['H_U']['wins'] += 1
            elif precisions["H"] < precisions["U"]:
                results['H_U']['losses'] += 1
            else:
                results['H_U']['ties'] += 1
            
            # O vs U comparison
            if precisions["O"] > precisions["U"]:
                results['O_U']['wins'] += 1
            elif precisions["O"] < precisions["U"]:
                results['O_U']['losses'] += 1
            else:
                results['O_U']['ties'] += 1
            
            # O vs H comparison
            if precisions["O"] > precisions["H"]:
                results['O_H']['wins'] += 1
            elif precisions["O"] < precisions["H"]:
                results['O_H']['losses'] += 1
            else:
                results['O_H']['ties'] += 1
        
        # Calculate summary statistics
        total_files = len(compiled_files)
        summary = {}
        
        for comparison, data in results.items():
            win_pct = (data['wins'] / total_files) * 100
            loss_pct = (data['losses'] / total_files) * 100
            
            summary[comparison] = {
                'win_count': data['wins'],
                'loss_count': data['losses'],
                'win_pct': win_pct,
                'loss_pct': loss_pct,
                'ties': data['ties']
            }
        
        return summary

    def calculate_pairwise_differences(self, compiled_files: List[str]) -> Dict:
        """Calculate pairwise differences for non-Any ratios to show the effect of user annotations."""
        print("Loading any-ratio data for difference analysis...")
        
        any_ratio_data = {}
        for scenario, file_path in self.any_ratio_files.items():
            any_ratio_data[scenario] = self.load_json_file(file_path)
        
        results = {
            'H_U': {'differences': [], 'wins': 0, 'losses': 0, 'ties': 0},
            'O_U': {'differences': [], 'wins': 0, 'losses': 0, 'ties': 0},
            'O_H': {'differences': [], 'wins': 0, 'losses': 0, 'ties': 0}
        }
        
        print(f"Analyzing pairwise differences for {len(compiled_files)} files...")
        
        for filename in compiled_files:
            # Get any percentages for each scenario
            original_any = any_ratio_data["original"].get(filename, {}).get("any_percentage", 100.0)
            partial_any = any_ratio_data["partial"].get(filename, {}).get("any_percentage", original_any)
            full_any = any_ratio_data["full"].get(filename, {}).get("any_percentage", original_any)
            
            # Calculate non-any ratios (100 - any_percentage)
            ratios = {
                "U": 100 - original_any,  # Untyped (original)
                "H": 100 - partial_any,   # Half-typed (partial)
                "O": 100 - full_any,      # Original types (full)
            }
            
            # Calculate differences (first scenario - second scenario)
            h_u_diff = ratios["H"] - ratios["U"]  # Half-typed - Untyped
            o_u_diff = ratios["O"] - ratios["U"]  # Original - Untyped
            o_h_diff = ratios["O"] - ratios["H"]  # Original - Half-typed
            
            # Store differences
            results['H_U']['differences'].append(h_u_diff)
            results['O_U']['differences'].append(o_u_diff)
            results['O_H']['differences'].append(o_h_diff)
            
            # Count wins/losses based on differences
            if h_u_diff > 0:
                results['H_U']['wins'] += 1
            elif h_u_diff < 0:
                results['H_U']['losses'] += 1
            else:
                results['H_U']['ties'] += 1
                
            if o_u_diff > 0:
                results['O_U']['wins'] += 1
            elif o_u_diff < 0:
                results['O_U']['losses'] += 1
            else:
                results['O_U']['ties'] += 1
                
            if o_h_diff > 0:
                results['O_H']['wins'] += 1
            elif o_h_diff < 0:
                results['O_H']['losses'] += 1
            else:
                results['O_H']['ties'] += 1
        
        # Calculate summary statistics
        total_files = len(compiled_files)
        summary = {}
        
        for comparison, data in results.items():
            win_pct = (data['wins'] / total_files) * 100
            loss_pct = (data['losses'] / total_files) * 100
            
            # Calculate average differences for wins and losses
            winning_diffs = [d for d in data['differences'] if d > 0]
            losing_diffs = [d for d in data['differences'] if d < 0]
            
            avg_win_diff = sum(winning_diffs) / len(winning_diffs) if winning_diffs else 0
            avg_loss_diff = sum(losing_diffs) / len(losing_diffs) if losing_diffs else 0
            
            summary[comparison] = {
                'win_count': data['wins'],
                'loss_count': data['losses'],
                'win_pct': win_pct,
                'loss_pct': loss_pct,
                'avg_win_diff': avg_win_diff,
                'avg_loss_diff': avg_loss_diff,
                'ties': data['ties']
            }
        
        return summary

    def format_latex_table_data(self, non_any_data: Dict, precision_data: Dict) -> str:
        """Format the analysis results into LaTeX table format."""
        model_name = self.llm_name.replace('-', '').replace('3', '3').title()
        if 'o3' in self.llm_name.lower():
            model_name = 'O3-mini'
        elif 'deepseek' in self.llm_name.lower():
            model_name = 'DeepSeek'
        elif 'claude' in self.llm_name.lower():
            model_name = 'Claude3'
        
        # Format non-Any ratio data (with averages)
        def format_non_any_cell(comparison):
            data = non_any_data[comparison]
            win_str = f"+{data['win_pct']:.1f} ({data['win_avg']:.1f})" if data['win_count'] > 0 else "+0.0 (0.0)"
            loss_str = f"-{data['loss_pct']:.1f} ({data['loss_avg']:.1f})" if data['loss_count'] > 0 else "-0.0 (0.0)"
            return f"{win_str} / {loss_str}"
        
        # Format precision data (without averages)
        def format_precision_cell(comparison):
            data = precision_data[comparison]
            win_str = f"+{data['win_pct']:.1f}" if data['win_count'] > 0 else "+0.0"
            loss_str = f"-{data['loss_pct']:.1f}" if data['loss_count'] > 0 else "-0.0"
            return f"{win_str} / {loss_str}"
        
        # Generate the table row
        latex_row = f"{model_name:<10} & "
        latex_row += f"{format_non_any_cell('H_U')} & {format_non_any_cell('O_U')} & {format_non_any_cell('O_H')} & "
        latex_row += f"{format_precision_cell('H_U')} & {format_precision_cell('O_U')} & {format_precision_cell('O_H')} & "
        latex_row += f"{format_non_any_cell('H_U')} & {format_non_any_cell('O_U')} & {format_non_any_cell('O_H')} \\\\"
        
        return latex_row

    def format_difference_table_data(self, difference_data: Dict, precision_data: Dict) -> str:
        """Format the difference analysis results into LaTeX table format."""
        model_name = self.llm_name.replace('-', '').replace('3', '3').title()
        if 'o3' in self.llm_name.lower():
            model_name = 'O3-mini'
        elif 'deepseek' in self.llm_name.lower():
            model_name = 'DeepSeek'
        elif 'claude' in self.llm_name.lower():
            model_name = 'Claude3'
        
        # Format difference data (with differences in parentheses)
        def format_difference_cell(comparison):
            data = difference_data[comparison]
            win_str = f"+{data['win_pct']:.1f} (+{data['avg_win_diff']:.1f})" if data['win_count'] > 0 else "+0.0 (+0.0)"
            loss_str = f"-{data['loss_pct']:.1f} ({data['avg_loss_diff']:.1f})" if data['loss_count'] > 0 else "-0.0 (0.0)"
            return f"{win_str} / {loss_str}"
        
        # Format precision data (without averages)
        def format_precision_cell(comparison):
            data = precision_data[comparison]
            win_str = f"+{data['win_pct']:.1f}" if data['win_count'] > 0 else "+0.0"
            loss_str = f"-{data['loss_pct']:.1f}" if data['loss_count'] > 0 else "-0.0"
            return f"{win_str} / {loss_str}"
        
        # Generate the table row
        latex_row = f"{model_name:<10} & "
        latex_row += f"{format_difference_cell('H_U')} & {format_difference_cell('O_U')} & {format_difference_cell('O_H')} & "
        latex_row += f"{format_precision_cell('H_U')} & {format_precision_cell('O_U')} & {format_precision_cell('O_H')} & "
        latex_row += f"{format_difference_cell('H_U')} & {format_difference_cell('O_U')} & {format_difference_cell('O_H')} \\\\"
        
        return latex_row

    def run_analysis(self) -> AnalysisResult:
        """Run the complete analysis."""
        print(f"=== {self.llm_name.upper()} Analysis ===\n")

        # Step 1: Find common compiled files
        compiled_files = self.find_common_compiled_files()
        if not compiled_files:
            print("No common compiled files found!")
            return None

        print(f"Analyzing {len(compiled_files)} files\n")

        # Step 2: Non-Any ratio analysis (compiled files only)
        print("=== Non-Any Ratio Analysis (Compiled Files Only) ===")
        non_any_ratios = self.calculate_non_any_ratios(compiled_files)
        O_ratio, U_ratio, H_ratio = non_any_ratios

        print(f"Results:")
        print(
            f"  Full (O) ratio: {O_ratio:.4f} ({O_ratio*100:.2f}%) - {self.non_any_file_counts['O']} files"
        )
        print(
            f"  Original (U) ratio: {U_ratio:.4f} ({U_ratio*100:.2f}%) - {self.non_any_file_counts['U']} files"
        )
        print(
            f"  Partial (H) ratio: {H_ratio:.4f} ({H_ratio*100:.2f}%) - {self.non_any_file_counts['H']} files"
        )
        print()

        # Step 2.5: Non-Any ratio analysis (all files)
        print("=== Non-Any Ratio Analysis (All Files) ===")
        non_any_ratios_all = self.calculate_non_any_ratios_all_files()
        O_ratio_all, U_ratio_all, H_ratio_all = non_any_ratios_all

        print(f"Results:")
        print(
            f"  Full (O) ratio: {O_ratio_all:.4f} ({O_ratio_all*100:.2f}%) - {self.non_any_file_counts_all['O']} files"
        )
        print(
            f"  Original (U) ratio: {U_ratio_all:.4f} ({U_ratio_all*100:.2f}%) - {self.non_any_file_counts_all['U']} files"
        )
        print(
            f"  Partial (H) ratio: {H_ratio_all:.4f} ({H_ratio_all*100:.2f}%) - {self.non_any_file_counts_all['H']} files"
        )
        print()

        # Step 3: Precision analysis
        print("=== Precision Analysis ===")
        precision_ratios = self.calculate_precision_ratios(compiled_files)
        O_prec, U_prec, H_prec = precision_ratios

        print(f"Results:")
        print(
            f"  Full (O) precision ratio: {O_prec:.4f} ({O_prec*100:.2f}%) - {self.precision_file_counts['O']} files"
        )
        print(
            f"  Original (U) precision ratio: {U_prec:.4f} ({U_prec*100:.2f}%) - {self.precision_file_counts['U']} files"
        )
        print(
            f"  Partial (H) precision ratio: {H_prec:.4f} ({H_prec*100:.2f}%) - {self.precision_file_counts['H']} files"
        )
        print()
        
        # Step 4: Organic gains/losses analysis
        print("=== Organic Gains/Losses Analysis ===")
        organic_results = self.analyze_organic_gains_losses(compiled_files)
        
        print(f"Results (parameter-level analysis):")
        for comparison, data in organic_results.items():
            gains_pct = (data['gains'] / data['total'] * 100) if data['total'] > 0 else 0
            losses_pct = (data['losses'] / data['total'] * 100) if data['total'] > 0 else 0
            neutral_pct = (data['neutral'] / data['total'] * 100) if data['total'] > 0 else 0
            net_gain = data['gains'] - data['losses']
            
            print(f"  {comparison}:")
            print(f"    Gains: {data['gains']} ({gains_pct:.1f}%)")
            print(f"    Losses: {data['losses']} ({losses_pct:.1f}%)")
            print(f"    Neutral: {data['neutral']} ({neutral_pct:.1f}%)")
            print(f"    Net gain: {net_gain} parameters")
            print(f"    Total parameters: {data['total']}")
            print()

        # Summary
        print("=== Summary ===")
        print(f"Total compiled files analyzed: {len(compiled_files)}")

        non_any_winner = (
            "Full"
            if O_ratio > max(U_ratio, H_ratio)
            else "Partial" if H_ratio > U_ratio else "Original"
        )
        non_any_winner_all = (
            "Full"
            if O_ratio_all > max(U_ratio_all, H_ratio_all)
            else "Partial" if H_ratio_all > U_ratio_all else "Original"
        )
        precision_winner = (
            "Full"
            if O_prec > max(U_prec, H_prec)
            else "Partial" if H_prec > U_prec else "Original"
        )

        print(f"Non-Any Ratio Winner (compiled files): {non_any_winner}")
        print(f"Non-Any Ratio Winner (all files): {non_any_winner_all}")
        print(f"Precision Winner: {precision_winner}")

        # Step 5: Pairwise comparisons for LaTeX table
        print("=== Pairwise Comparisons for LaTeX Table ===")
        non_any_pairwise = self.calculate_pairwise_non_any_comparisons(compiled_files)
        precision_pairwise = self.calculate_pairwise_precision_comparisons(compiled_files)
        
        # Print detailed pairwise results
        print("Non-Any Ratio Pairwise Comparisons:")
        for comparison, data in non_any_pairwise.items():
            print(f"  {comparison}: {data['win_count']} wins ({data['win_pct']:.1f}%), {data['loss_count']} losses ({data['loss_pct']:.1f}%), {data['ties']} ties")
            if data['win_count'] > 0:
                print(f"    Average winning ratio: {data['win_avg']:.2f}%")
            if data['loss_count'] > 0:
                print(f"    Average losing ratio: {data['loss_avg']:.2f}%")
        
        print("\nPrecision Pairwise Comparisons:")
        for comparison, data in precision_pairwise.items():
            print(f"  {comparison}: {data['win_count']} wins ({data['win_pct']:.1f}%), {data['loss_count']} losses ({data['loss_pct']:.1f}%), {data['ties']} ties")
        
        # Generate LaTeX table row
        latex_row = self.format_latex_table_data(non_any_pairwise, precision_pairwise)
        print(f"\nLaTeX Table Row:")
        print(latex_row)

        return AnalysisResult(
            llm_name=self.llm_name,
            total_files=len(compiled_files),
            non_any_ratios=non_any_ratios,
            non_any_ratios_all=non_any_ratios_all,
            precision_ratios=precision_ratios,
            compiled_files=compiled_files,
        )


def analyze_llm(llm_name: str) -> AnalysisResult:
    """Analyze a specific LLM and return results."""
    analyzer = LLMAnalyzer(llm_name)
    return analyzer.run_analysis()


def compare_llms(llm_names: List[str]) -> Dict[str, AnalysisResult]:
    """Compare multiple LLMs."""
    results = {}
    for llm_name in llm_names:
        print(f"\n{'='*60}")
        results[llm_name] = analyze_llm(llm_name)
        print(f"{'='*60}")

    return results


def generate_latex_table(llm_names: List[str]) -> str:
    """Generate complete LaTeX table for all specified LLMs."""
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\scriptsize")
    latex_lines.append("\\caption{Precise Output and Precise Input Correlation.}")
    latex_lines.append("\\label{tab:precise-distribution}")
    latex_lines.append("\\resizebox{\\linewidth}{!}{%")
    latex_lines.append("\\begin{tabular}{l | c c c | c c c | c c c}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\multirow{2}{*}{Model} & \\multicolumn{3}{c|}{WT \\& highest ratio} & \\multicolumn{3}{c|}{WT \\& most precise} & \\multicolumn{3}{c}{Highest ratio} \\\\")
    latex_lines.append("& $H_U$ & $O_U$ & $O_H$ & $H_U$ & $O_U$ & $O_H$ & $H_U$ & $O_U$ & $O_H$ \\\\")
    latex_lines.append("\\midrule")
    
    for llm_name in llm_names:
        analyzer = LLMAnalyzer(llm_name)
        compiled_files = analyzer.find_common_compiled_files()
        if compiled_files:
            non_any_pairwise = analyzer.calculate_pairwise_non_any_comparisons(compiled_files)
            precision_pairwise = analyzer.calculate_pairwise_precision_comparisons(compiled_files)
            latex_row = analyzer.format_latex_table_data(non_any_pairwise, precision_pairwise)
            latex_lines.append(latex_row)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)


def generate_difference_latex_table(llm_names: List[str]) -> str:
    """Generate complete LaTeX table showing differences for all specified LLMs."""
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\scriptsize")
    latex_lines.append("\\caption{Effect of User Annotations on Precision Results (Differences).}")
    latex_lines.append("\\label{tab:user-annotation-effect}")
    latex_lines.append("\\resizebox{\\linewidth}{!}{%")
    latex_lines.append("\\begin{tabular}{l | c c c | c c c | c c c}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\multirow{2}{*}{Model} & \\multicolumn{3}{c|}{WT \\& highest ratio} & \\multicolumn{3}{c|}{WT \\& most precise} & \\multicolumn{3}{c}{Highest ratio} \\\\")
    latex_lines.append("& $H_U$ & $O_U$ & $O_H$ & $H_U$ & $O_U$ & $O_H$ & $H_U$ & $O_U$ & $O_H$ \\\\")
    latex_lines.append("\\midrule")
    
    for llm_name in llm_names:
        analyzer = LLMAnalyzer(llm_name)
        compiled_files = analyzer.find_common_compiled_files()
        if compiled_files:
            difference_data = analyzer.calculate_pairwise_differences(compiled_files)
            precision_data = analyzer.calculate_pairwise_precision_comparisons(compiled_files)
            latex_row = analyzer.format_difference_table_data(difference_data, precision_data)
            latex_lines.append(latex_row)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python llm_analysis.py [claude|deepseek|o3-mini] [--latex|--difference]")
        print("Example: python llm_analysis.py claude")
        print("Example: python llm_analysis.py claude deepseek o3-mini --latex")
        print("Example: python llm_analysis.py claude deepseek o3-mini --difference")
        return

    # Check for flags
    latex_mode = "--latex" in sys.argv
    difference_mode = "--difference" in sys.argv
    llm_names = [arg for arg in sys.argv[1:] if arg not in ["--latex", "--difference"]]

    if latex_mode:
        # Generate LaTeX table
        print("Generating LaTeX table...")
        latex_table = generate_latex_table(llm_names)
        print("\n" + "="*80)
        print("LATEX TABLE OUTPUT:")
        print("="*80)
        print(latex_table)
        print("="*80)
    elif difference_mode:
        # Generate difference LaTeX table
        print("Generating difference LaTeX table...")
        difference_table = generate_difference_latex_table(llm_names)
        print("\n" + "="*80)
        print("DIFFERENCE LATEX TABLE OUTPUT:")
        print("="*80)
        print(difference_table)
        print("="*80)
    elif len(llm_names) == 1:
        # Single LLM analysis
        result = analyze_llm(llm_names[0])
        if result:
            print(f"\n=== Final Results for {result.llm_name.upper()} ===")
            print(f"Compiled files analyzed: {result.total_files}")
            print(f"Non-Any ratios (compiled files) (O, U, H): {result.non_any_ratios}")
            print(f"Non-Any ratios (all files) (O, U, H): {result.non_any_ratios_all}")
            print(f"Precision ratios (O, U, H): {result.precision_ratios}")
    else:
        # Multiple LLM comparison
        results = compare_llms(llm_names)

        print(f"\n{'='*80}")
        print("=== COMPARISON SUMMARY ===")
        print(f"{'='*80}")

        for llm_name, result in results.items():
            if result:
                print(f"\n{llm_name.upper()}:")
                print(f"  Compiled files: {result.total_files}")
                print(
                    f"  Non-Any ratios (compiled) (O, U, H): {[f'{r:.3f}' for r in result.non_any_ratios]}"
                )
                print(
                    f"  Non-Any ratios (all files) (O, U, H): {[f'{r:.3f}' for r in result.non_any_ratios_all]}"
                )
                print(
                    f"  Precision ratios (O, U, H): {[f'{r:.3f}' for r in result.precision_ratios]}"
                )


if __name__ == "__main__":
    main()
