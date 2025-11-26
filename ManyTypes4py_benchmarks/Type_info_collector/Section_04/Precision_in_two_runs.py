import json
import os
import csv
import re
from collections import defaultdict


def load_type_info(file_path):
    """Load type information from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_type_precision_score(type_str):
    """Calculate precision score for a type annotation.
    Higher score = more precise type."""
    if not isinstance(type_str, str):
        return 0
    
    type_str = type_str.strip().lower()
    
    # Base precision scores
    if type_str == "any":
        return 0
    elif type_str in ["object", "typing.any"]:
        return 1
    elif type_str in ["str", "int", "float", "bool", "bytes", "complex"]:
        return 10
    elif type_str in ["list", "dict", "set", "tuple"]:
        return 5
    elif type_str in ["none", "nonetype"]:
        return 1
    
    # Handle generic types with type parameters
    # List[Any] = 5, List[str] = 15, List[List[str]] = 25
    if type_str.startswith("list["):
        inner_type = type_str[5:-1]  # Remove "list[" and "]"
        inner_score = get_type_precision_score(inner_type)
        return 10 + inner_score
    
    elif type_str.startswith("dict["):
        # dict[str, any] = 5, dict[str, str] = 15
        if "," in type_str:
            key_type = type_str[5:type_str.find(",")]
            value_type = type_str[type_str.find(",")+1:-1]
            key_score = get_type_precision_score(key_type)
            value_score = get_type_precision_score(value_type)
            return 10 + key_score + value_score
        else:
            return 5
    
    elif type_str.startswith("set["):
        inner_type = type_str[4:-1]
        inner_score = get_type_precision_score(inner_type)
        return 10 + inner_score
    
    elif type_str.startswith("tuple["):
        # tuple[str, int] = 20, tuple[any, any] = 10
        if "," in type_str:
            inner_types = type_str[6:-1].split(",")
            total_score = 10
            for inner_type in inner_types:
                total_score += get_type_precision_score(inner_type.strip())
            return total_score
        else:
            inner_type = type_str[6:-1]
            return 10 + get_type_precision_score(inner_type)
    
    elif type_str.startswith("union["):
        # union[str, int] = 8, union[any, any] = 2
        if "," in type_str:
            inner_types = type_str[6:-1].split(",")
            total_score = 2
            for inner_type in inner_types:
                total_score += get_type_precision_score(inner_type.strip())
            return total_score
        else:
            inner_type = type_str[6:-1]
            return 2 + get_type_precision_score(inner_type)
    
    elif type_str.startswith("optional["):
        inner_type = type_str[9:-1]
        return 8 + get_type_precision_score(inner_type)
    
    # Handle other common types
    elif "typing." in type_str:
        return 8
    
    # Default for unknown types
    return 5


def extract_common_type_slots(run1_data, run2_data):
    """Extract common parameter and return type slots between two runs."""
    run1_slots = {}
    run2_slots = {}
    
    # Extract slots from run1
    for filename, functions in run1_data.items():
        if isinstance(functions, dict):
            for func_name, func_data in functions.items():
                if isinstance(func_data, list):
                    for param in func_data:
                        if isinstance(param, dict):
                            category = param.get("category", "")
                            param_name = param.get("name", "")
                            param_types = param.get("type", [])
                            
                            if isinstance(param_types, list) and len(param_types) > 0:
                                type_str = param_types[0]
                                if isinstance(type_str, str) and type_str.strip():
                                    slot_key = f"{filename}:{func_name}:{category}:{param_name}"
                                    run1_slots[slot_key] = type_str.strip()
    
    # Extract slots from run2
    for filename, functions in run2_data.items():
        if isinstance(functions, dict):
            for func_name, func_data in functions.items():
                if isinstance(func_data, list):
                    for param in func_data:
                        if isinstance(param, dict):
                            category = param.get("category", "")
                            param_name = param.get("name", "")
                            param_types = param.get("type", [])
                            
                            if isinstance(param_types, list) and len(param_types) > 0:
                                type_str = param_types[0]
                                if isinstance(type_str, str) and type_str.strip():
                                    slot_key = f"{filename}:{func_name}:{category}:{param_name}"
                                    run2_slots[slot_key] = type_str.strip()
    
    # Find common slots
    common_slots = set(run1_slots.keys()) & set(run2_slots.keys())
    
    return common_slots, run1_slots, run2_slots


def analyze_any_precision(common_slots, run1_slots, run2_slots):
    """Analyze Any precision for common slots between two runs."""
    any_both = 0
    any_run1_only = 0
    any_run2_only = 0
    any_neither = 0
    
    for slot in common_slots:
        run1_type = run1_slots[slot].lower()
        run2_type = run2_slots[slot].lower()
        
        run1_is_any = run1_type == "any"
        run2_is_any = run2_type == "any"
        
        if run1_is_any and run2_is_any:
            any_both += 1
        elif run1_is_any and not run2_is_any:
            any_run1_only += 1
        elif not run1_is_any and run2_is_any:
            any_run2_only += 1
        else:  # neither is any
            any_neither += 1
    
    return any_both, any_run1_only, any_run2_only, any_neither


def analyze_precision_comparison(common_slots, run1_slots, run2_slots):
    """Analyze precision comparison between two runs."""
    both_any = 0
    run1_more_precise = 0
    run2_more_precise = 0
    both_static = 0
    
    for slot in common_slots:
        run1_type = run1_slots[slot]
        run2_type = run2_slots[slot]
        
        run1_score = get_type_precision_score(run1_type)
        run2_score = get_type_precision_score(run2_type)
        
        if run1_score == 0 and run2_score == 0:
            both_any += 1
        elif run1_score > run2_score:
            run1_more_precise += 1
        elif run2_score > run1_score:
            run2_more_precise += 1
        else:  # equal scores (both static)
            both_static += 1
    
    return both_any, run1_more_precise, run2_more_precise, both_static


def analyze_by_category(common_slots, run1_slots, run2_slots):
    """Analyze Any precision separately for parameters and return types."""
    param_slots = [slot for slot in common_slots if slot.split(":")[2] == "arg"]
    return_slots = [slot for slot in common_slots if slot.split(":")[2] == "return"]
    
    param_results = analyze_any_precision(param_slots, run1_slots, run2_slots)
    return_results = analyze_any_precision(return_slots, run1_slots, run2_slots)
    
    return param_results, return_results


def analyze_precision_by_category(common_slots, run1_slots, run2_slots):
    """Analyze precision comparison separately for parameters and return types."""
    param_slots = [slot for slot in common_slots if slot.split(":")[2] == "arg"]
    return_slots = [slot for slot in common_slots if slot.split(":")[2] == "return"]
    
    param_results = analyze_precision_comparison(param_slots, run1_slots, run2_slots)
    return_results = analyze_precision_comparison(return_slots, run1_slots, run2_slots)
    
    return param_results, return_results


def main():
    # Define model pairs (1st and 2nd runs)
    model_pairs = {
        "GPT35": {
            "run1": "../Type_info_LLMS/Type_info_gpt35_1st_run_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_gpt35_2nd_run_benchmarks.json"
        },
        "GPT4o": {
            "run1": "../Type_info_LLMS/Type_info_gpt4o_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_gpt4o_2nd_run_benchmarks.json"
        },
        "O1-mini": {
            "run1": "../Type_info_LLMS/Type_info_o1_mini_1st_run_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_o1_mini_2nd_run_benchmarks.json"
        },
        "O3-mini": {
            "run1": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_o3_mini_2nd_run_benchmarks.json"
        },
        "DeepSeek": {
            "run1": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_deep_seek_2nd_run_benchmarks.json"
        },
        "Claude3-Sonnet": {
            "run1": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",
            "run2": "../Type_info_LLMS/Type_info_claude3_sonnet_2nd_run_benchmarks.json"
        }
    }
    
    results = {}
    
    for model_name, file_paths in model_pairs.items():
        # Load data for both runs
        run1_data = load_type_info(file_paths["run1"])
        run2_data = load_type_info(file_paths["run2"])
        
        if not run1_data or not run2_data:
            continue
        
        # Extract common slots
        common_slots, run1_slots, run2_slots = extract_common_type_slots(run1_data, run2_data)
        
        if not common_slots:
            continue
        
        # Analyze overall Any precision
        any_both, any_run1_only, any_run2_only, any_neither = analyze_any_precision(
            common_slots, run1_slots, run2_slots
        )
        
        # Analyze overall precision comparison
        both_any, run1_more_precise, run2_more_precise, both_static = analyze_precision_comparison(
            common_slots, run1_slots, run2_slots
        )
        
        # Analyze by category (Any precision)
        param_results, return_results = analyze_by_category(common_slots, run1_slots, run2_slots)
        
        # Analyze by category (precision comparison)
        param_precision_results, return_precision_results = analyze_precision_by_category(common_slots, run1_slots, run2_slots)
        
        total_common = len(common_slots)
        param_common = len([slot for slot in common_slots if slot.split(":")[2] == "arg"])
        return_common = len([slot for slot in common_slots if slot.split(":")[2] == "return"])
        
        results[model_name] = {
            "total_common": total_common,
            "any_both": any_both,
            "any_run1_only": any_run1_only,
            "any_run2_only": any_run2_only,
            "any_neither": any_neither,
            "both_any": both_any,
            "run1_more_precise": run1_more_precise,
            "run2_more_precise": run2_more_precise,
            "both_static": both_static,
            "param_common": param_common,
            "param_any_both": param_results[0],
            "param_any_run1_only": param_results[1],
            "param_any_run2_only": param_results[2],
            "param_any_neither": param_results[3],
            "param_both_any": param_precision_results[0],
            "param_run1_more_precise": param_precision_results[1],
            "param_run2_more_precise": param_precision_results[2],
            "param_both_static": param_precision_results[3],
            "return_common": return_common,
            "return_any_both": return_results[0],
            "return_any_run1_only": return_results[1],
            "return_any_run2_only": return_results[2],
            "return_any_neither": return_results[3],
            "return_both_any": return_precision_results[0],
            "return_run1_more_precise": return_precision_results[1],
            "return_run2_more_precise": return_precision_results[2],
            "return_both_static": return_precision_results[3]
        }
    
    # Print original Any-based summary table
    print(f"{'Model':<15} {'Total':<8} {'Any Both':<10} {'Any 1st':<10} {'Any 2nd':<10} {'Any Neither':<12}")
    print("-" * 80)
    
    for model_name, data in results.items():
        print(f"{model_name:<15} {data['total_common']:<8} {data['any_both']:<10} "
              f"{data['any_run1_only']:<10} {data['any_run2_only']:<10} {data['any_neither']:<12}")
    
    # Print precision-based summary table
    print(f"\n{'Model':<15} {'Total':<8} {'Both Any':<10} {'1st Precise':<12} {'2nd Precise':<12} {'Both Static':<12}")
    print("-" * 80)
    
    for model_name, data in results.items():
        print(f"{model_name:<15} {data['total_common']:<8} {data['both_any']:<10} "
              f"{data['run1_more_precise']:<12} {data['run2_more_precise']:<12} {data['both_static']:<12}")


if __name__ == "__main__":
    main()
