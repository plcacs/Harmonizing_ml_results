import json
import csv
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import numpy as np

# Configuration for LLMs including Human baseline
LLM_CONFIGS = {
    "gpt-3.5": {
        "type_info_path": "../Type_info_LLMS/Type_info_gpt35_2nd_run_benchmarks.json",
    },
    "gpt-4o": {
        "type_info_path": "../Type_info_LLMS/Type_info_gpt4o_benchmarks.json",
    },
    "o1-mini": {
        "type_info_path": "../Type_info_LLMS/Type_info_o1_mini_benchmarks.json",
    },
    "o3-mini": {
        "type_info_path": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
    },
    "deepseek": {
        "type_info_path": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
    },
    "claude3-sonnet": {
        "type_info_path": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",
    },
    "Human": {
        "type_info_path": "../Type_info_LLMS/Type_info_original_files.json",
    }
}

def load_json(path: str) -> Dict:
    """Load JSON file with error handling."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}

def normalize_type(type_str: str) -> str:
    """Normalize type string for comparison."""
    if not isinstance(type_str, str):
        return ""
    
    # Remove common prefixes and normalize
    type_str = type_str.strip().lower()
    type_str = re.sub(r'^typing\.', '', type_str)
    type_str = re.sub(r'^tp\.', '', type_str)
    type_str = re.sub(r'^t\.', '', type_str)
    
    return type_str

def normalize_union_types(type_str: str) -> str:
    """Normalize different union forms to a standard representation."""
    normalized = type_str.strip().lower()
    
    # Handle Optional[int] -> Union[int, None]
    if normalized.startswith('optional[') and normalized.endswith(']'):
        inner_type = normalized[9:-1]  # Extract content inside Optional[...]
        return f"union[{inner_type}, none]"
    
    # Handle int | None -> Union[int, None]  
    if '|' in normalized:
        parts = [p.strip() for p in normalized.split('|')]
        # Sort parts for consistent ordering
        parts.sort()
        return f"union[{','.join(parts)}]"
    
    # Handle Union[int, str] -> Union[int, str] (already in Union format)
    if normalized.startswith('union[') and normalized.endswith(']'):
        inner_content = normalized[6:-1]  # Extract content inside Union[...]
        parts = [p.strip() for p in inner_content.split(',')]
        # Sort parts for consistent ordering
        parts.sort()
        return f"union[{','.join(parts)}]"
    
    return normalized

def categorize_type(type_str: str) -> str:
    """Categorize type into broad categories."""
    if not type_str:
        return "empty"
    
    normalized = normalize_type(type_str)
    
    # Primitive types
    if normalized in ['str', 'int', 'float', 'bool', 'bytes', 'complex']:
        return "primitive"
    
    # Container types
    if any(container in normalized for container in ['list', 'dict', 'set', 'tuple']):
        return "container"
    
    # Generic types (type variables like T, KT, VT)
    if re.search(r'\b[A-Z][A-Z0-9_]*\b', normalized) and len(normalized) <= 3:
        return "generic"
    
    # Union types
    if 'union' in normalized or '|' in normalized:
        return "union"
    
    # Custom types (likely class names)
    if re.match(r'^[A-Z][a-zA-Z0-9_]*$', type_str):
        return "custom"
    
    return "other"

def are_types_semantically_similar(human_type: str, llm_type: str) -> bool:
    """Check if two types are semantically similar."""
    if not human_type or not llm_type:
        return False
    
    human_norm = normalize_type(human_type)
    llm_norm = normalize_type(llm_type)
    
    # Normalize union types for comparison
    human_union_norm = normalize_union_types(human_norm)
    llm_union_norm = normalize_union_types(llm_norm)
    
    # Exact match (including normalized union forms)
    if human_norm == llm_norm or human_union_norm == llm_union_norm:
        return True
    
    # Both are Any-like
    if human_norm in ['any', 'object'] and llm_norm in ['any', 'object']:
        return True
    
    # Both are primitive types (treat as similar for analysis)
    human_cat = categorize_type(human_type)
    llm_cat = categorize_type(llm_type)
    
    if human_cat == "primitive" and llm_cat == "primitive":
        return True
    
    # Both are containers with similar structure
    if human_cat == "container" and llm_cat == "container":
        # Extract base container type
        human_base = re.search(r'(list|dict|set|tuple)', human_norm)
        llm_base = re.search(r'(list|dict|set|tuple)', llm_norm)
        if human_base and llm_base and human_base.group(1) == llm_base.group(1):
            return True
    
    return False

def analyze_parameter_agreement(human_data: Dict, llm_data: Dict, llm_name: str) -> Dict:
    """Analyze parameter-level agreement between Human and LLM."""
    total_common_params = 0
    agreement_count = 0
    disagreement_count = 0
    
    # Category-specific counts
    category_agreements = {
        "primitive": {"total": 0, "agreed": 0},
        "container": {"total": 0, "agreed": 0},
        "generic": {"total": 0, "agreed": 0},
        "union": {"total": 0, "agreed": 0},
        "custom": {"total": 0, "agreed": 0},
        "other": {"total": 0, "agreed": 0}
    }
    
    # Find common files
    common_files = set(human_data.keys()) & set(llm_data.keys())
    
    for filename in common_files:
        human_functions = human_data[filename]
        llm_functions = llm_data[filename]
        
        # Find common functions
        common_funcs = set(human_functions.keys()) & set(llm_functions.keys())
        
        for func_name in common_funcs:
            human_params = human_functions[func_name]
            llm_params = llm_functions[func_name]
            
            # Create lookup dictionaries
            human_dict = {}
            for param in human_params:
                if isinstance(param, dict):
                    name = param.get("name", "")
                    category = param.get("category", "")
                    # Skip method receiver parameter named "self"
                    if isinstance(name, str) and category == "arg" and name.strip().lower() == "self":
                        continue
                    key = (category, name)
                    human_dict[key] = param
            
            llm_dict = {}
            for param in llm_params:
                if isinstance(param, dict):
                    name = param.get("name", "")
                    category = param.get("category", "")
                    # Skip method receiver parameter named "self"
                    if isinstance(name, str) and category == "arg" and name.strip().lower() == "self":
                        continue
                    key = (category, name)
                    llm_dict[key] = param
            
            # Find common parameters
            common_keys = set(human_dict.keys()) & set(llm_dict.keys())
            
            for key in common_keys:
                human_param = human_dict[key]
                llm_param = llm_dict[key]
                
                human_type = human_param.get("type", [""])[0] if human_param.get("type") else ""
                llm_type = llm_param.get("type", [""])[0] if llm_param.get("type") else ""
                
                # Skip if either type is empty or blank
                if not human_type or not llm_type or human_type.strip() == "" or llm_type.strip() == "":
                    continue
                
                total_common_params += 1
                
                # Categorize human type
                human_category = categorize_type(human_type)
                if human_category in category_agreements:
                    category_agreements[human_category]["total"] += 1
                
                # Check agreement
                if are_types_semantically_similar(human_type, llm_type):
                    agreement_count += 1
                    if human_category in category_agreements:
                        category_agreements[human_category]["agreed"] += 1
                else:
                    disagreement_count += 1
    
    # Calculate percentages
    agreement_percentage = (agreement_count / total_common_params * 100) if total_common_params > 0 else 0
    disagreement_percentage = (disagreement_count / total_common_params * 100) if total_common_params > 0 else 0
    
    # Calculate category-specific percentages
    category_percentages = {}
    for category, counts in category_agreements.items():
        if counts["total"] > 0:
            category_percentages[category] = (counts["agreed"] / counts["total"] * 100)
        else:
            category_percentages[category] = 0.0
    
    return {
        "llm_name": llm_name,
        "total_common_params": total_common_params,
        "agreement_count": agreement_count,
        "agreement_percentage": agreement_percentage,
        "disagreement_count": disagreement_count,
        "disagreement_percentage": disagreement_percentage,
        "category_agreements": category_percentages
    }

def analyze_file_level_agreement(human_data: Dict, llm_data: Dict, llm_name: str) -> Dict:
    """Analyze file-level agreement metrics."""
    common_files = set(human_data.keys()) & set(llm_data.keys())
    total_files = len(common_files)
    
    file_agreements = []
    
    for filename in common_files:
        human_functions = human_data[filename]
        llm_functions = llm_data[filename]
        
        # Count parameters in each file (exclude parameters named "self")
        def count_non_self(params_list):
            total = 0
            for func_params in params_list.values():
                if not isinstance(func_params, list):
                    continue
                for p in func_params:
                    if not isinstance(p, dict):
                        continue
                    name = p.get("name", "")
                    category = p.get("category", "")
                    if isinstance(name, str) and category == "arg" and name.strip().lower() == "self":
                        continue
                    total += 1
            return total

        human_params = count_non_self(human_functions)
        llm_params = count_non_self(llm_functions)
        
        # Calculate file-level agreement ratio
        if human_params > 0 and llm_params > 0:
            agreement_ratio = min(human_params, llm_params) / max(human_params, llm_params)
            file_agreements.append(agreement_ratio)
    
    avg_file_agreement = np.mean(file_agreements) if file_agreements else 0
    
    return {
        "llm_name": llm_name,
        "total_common_files": total_files,
        "avg_file_agreement": avg_file_agreement * 100
    }

def main():
    """Main analysis function."""
    print("Loading data...")
    
    # Load all data
    all_data = {}
    for llm_name, config in LLM_CONFIGS.items():
        data = load_json(config["type_info_path"])
        all_data[llm_name] = data
        print(f"Loaded {llm_name}: {len(data)} files")
    
    human_data = all_data["Human"]
    
    print("\n" + "="*80)
    print("PRECISION AGREEMENT ANALYSIS: LLMs vs Human")
    print("="*80)
    
    # Analyze each LLM
    results = []
    
    for llm_name in ["gpt-3.5", "gpt-4o", "o1-mini", "o3-mini", "deepseek", "claude3-sonnet"]:
        if llm_name not in all_data:
            continue
            
        print(f"\nAnalyzing {llm_name}...")
        
        # Parameter-level analysis
        param_analysis = analyze_parameter_agreement(human_data, all_data[llm_name], llm_name)
        
        # File-level analysis
        file_analysis = analyze_file_level_agreement(human_data, all_data[llm_name], llm_name)
        
        # Combine results
        result = {
            "LLM": llm_name,
            "Common_Parameters": param_analysis["total_common_params"],
            "Agreement_Count": param_analysis["agreement_count"],
            "Agreement_Percentage": param_analysis["agreement_percentage"],
            "Disagreement_Count": param_analysis["disagreement_count"],
            "Disagreement_Percentage": param_analysis["disagreement_percentage"],
            "Primitive_Agreement": param_analysis["category_agreements"].get("primitive", 0),
            "Container_Agreement": param_analysis["category_agreements"].get("container", 0),
            "Generic_Agreement": param_analysis["category_agreements"].get("generic", 0),
            "Union_Agreement": param_analysis["category_agreements"].get("union", 0),
            "Custom_Agreement": param_analysis["category_agreements"].get("custom", 0),
            "File_Level_Agreement": file_analysis["avg_file_agreement"]
        }
        
        results.append(result)
        
        print(f"  Common Parameters: {param_analysis['total_common_params']:,}")
        print(f"  Agreement: {param_analysis['agreement_count']:,} ({param_analysis['agreement_percentage']:.1f}%)")
        print(f"  Disagreement: {param_analysis['disagreement_count']:,} ({param_analysis['disagreement_percentage']:.1f}%)")
        print(f"  File-level Agreement: {file_analysis['avg_file_agreement']:.1f}%")
    
    # Analyze union of top 3 models (Claude 3 Sonnet + GPT-4o + DeepSeek)
    print(f"\nAnalyzing Union of Top 3 Models (Claude 3 Sonnet + GPT-4o + DeepSeek)...")
    
    # Create union dataset
    top3_models = ["claude3-sonnet", "gpt-4o", "deepseek"]
    union_data = {}
    
    # Find files that exist in human data and at least one of the 3 models
    common_files = set(human_data.keys())
    model_files = set()
    for model in top3_models:
        if model in all_data:
            model_files = model_files | set(all_data[model].keys())
    common_files = common_files & model_files
    
    # For each common file, take the best annotation for each parameter
    for filename in common_files:
        union_data[filename] = {}
        human_file_data = human_data[filename]
        
        for func_name in human_file_data.keys():
            if func_name not in union_data[filename]:
                union_data[filename][func_name] = []
            
            # For each parameter, select the best annotation among the 3 models
            human_params = human_file_data[func_name]
            if not isinstance(human_params, list):
                continue
                
            for human_param in human_params:
                if not isinstance(human_param, dict):
                    continue
                    
                param_name = human_param.get("name", "")
                param_category = human_param.get("category", "")
                human_type = human_param.get("type", [""])[0] if human_param.get("type") else ""

                # Skip method receiver parameter named "self"
                if (
                    isinstance(param_name, str)
                    and param_category == "arg"
                    and param_name.strip().lower() == "self"
                ):
                    continue
                
                # Find the best annotation for this parameter among the 3 models
                best_annotation = None
                best_agreement = False
                
                # First, try to find an annotation that agrees with human
                for model in top3_models:
                    if model in all_data and filename in all_data[model] and func_name in all_data[model][filename]:
                        model_params = all_data[model][filename][func_name]
                        if isinstance(model_params, list):
                            for model_param in model_params:
                                if isinstance(model_param, dict):
                                    if (model_param.get("name") == param_name and 
                                        model_param.get("category") == param_category):
                                        model_type = model_param.get("type", [""])[0] if model_param.get("type") else ""
                                        # Check if this annotation agrees with human
                                        if are_types_semantically_similar(human_type, model_type):
                                            best_annotation = model_param
                                            best_agreement = True
                                            break
                            if best_agreement:
                                break
                
                # If no agreeing annotation found, take the first available one from any model
                if best_annotation is None:
                    for model in top3_models:
                        if model in all_data and filename in all_data[model] and func_name in all_data[model][filename]:
                            model_params = all_data[model][filename][func_name]
                            if isinstance(model_params, list):
                                for model_param in model_params:
                                    if isinstance(model_param, dict):
                                        if (model_param.get("name") == param_name and 
                                            model_param.get("category") == param_category):
                                            best_annotation = model_param
                                            break
                                if best_annotation:
                                    break
                
                # Only add if an actual model annotation was found
                # Do not fall back to human annotation to avoid inflating agreement metrics
                if best_annotation:
                    union_data[filename][func_name].append(best_annotation)
    
    # Analyze union performance
    union_param_analysis = analyze_parameter_agreement(human_data, union_data, "Union of Top 3")
    union_file_analysis = analyze_file_level_agreement(human_data, union_data, "Union of Top 3")
    
    # Add union results
    union_result = {
        "LLM": "Union of Top 3",
        "Common_Parameters": union_param_analysis["total_common_params"],
        "Agreement_Count": union_param_analysis["agreement_count"],
        "Agreement_Percentage": union_param_analysis["agreement_percentage"],
        "Disagreement_Count": union_param_analysis["disagreement_count"],
        "Disagreement_Percentage": union_param_analysis["disagreement_percentage"],
        "Primitive_Agreement": union_param_analysis["category_agreements"].get("primitive", 0),
        "Container_Agreement": union_param_analysis["category_agreements"].get("container", 0),
        "Generic_Agreement": union_param_analysis["category_agreements"].get("generic", 0),
        "Union_Agreement": union_param_analysis["category_agreements"].get("union", 0),
        "Custom_Agreement": union_param_analysis["category_agreements"].get("custom", 0),
        "File_Level_Agreement": union_file_analysis["avg_file_agreement"]
    }
    
    results.append(union_result)
    
    print(f"  Common Parameters: {union_param_analysis['total_common_params']:,}")
    print(f"  Agreement: {union_param_analysis['agreement_count']:,} ({union_param_analysis['agreement_percentage']:.1f}%)")
    print(f"  Disagreement: {union_param_analysis['disagreement_count']:,} ({union_param_analysis['disagreement_percentage']:.1f}%)")
    print(f"  File-level Agreement: {union_file_analysis['avg_file_agreement']:.1f}%")
    
    # Save results to CSV
    output_file = "precision_agreement_results.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "LLM", "Common_Parameters", "Agreement_Count", "Agreement_Percentage",
            "Disagreement_Count", "Disagreement_Percentage", "Primitive_Agreement",
            "Container_Agreement", "Generic_Agreement", "Union_Agreement", 
            "Custom_Agreement", "File_Level_Agreement"
        ])
        
        for result in results:
            writer.writerow([
                result["LLM"], result["Common_Parameters"], result["Agreement_Count"],
                f"{result['Agreement_Percentage']:.2f}", result["Disagreement_Count"],
                f"{result['Disagreement_Percentage']:.2f}", f"{result['Primitive_Agreement']:.2f}",
                f"{result['Container_Agreement']:.2f}", f"{result['Generic_Agreement']:.2f}",
                f"{result['Union_Agreement']:.2f}", f"{result['Custom_Agreement']:.2f}",
                f"{result['File_Level_Agreement']:.2f}"
            ])
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary table
    print("\n" + "="*120)
    print("SUMMARY TABLE")
    print("="*120)
    print(f"{'LLM':<20} {'Common':<8} {'Agree':<8} {'Agree%':<8} {'Disagree':<10} {'Disagree%':<10} {'Primitive':<10} {'Container':<10} {'Generic':<10} {'Union':<8} {'Custom':<8} {'File%':<8}")
    print("-"*125)
    
    for result in results:
        print(f"{result['LLM']:<20} {result['Common_Parameters']:<8,} {result['Agreement_Count']:<8,} "
              f"{result['Agreement_Percentage']:<8.1f} {result['Disagreement_Count']:<10,} "
              f"{result['Disagreement_Percentage']:<10.1f} {result['Primitive_Agreement']:<10.1f} "
              f"{result['Container_Agreement']:<10.1f} {result['Generic_Agreement']:<10.1f} "
              f"{result['Union_Agreement']:<8.1f} {result['Custom_Agreement']:<8.1f} "
              f"{result['File_Level_Agreement']:<8.1f}")

if __name__ == "__main__":
    main()
