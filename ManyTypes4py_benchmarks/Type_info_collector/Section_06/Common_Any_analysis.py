import json
import os

def analyze_common_any_blank_types():
    """Find parameters and return types that are Any/blank in ALL 3 LLMs"""
    
    print("Starting common Any/blank analysis...")
    
    # Load untyped mypy results to filter successful files
    untyped_mypy_path = "../../mypy_results/mypy_outputs/mypy_results_untyped_with_errors.json"
    with open(untyped_mypy_path, 'r') as f:
        untyped_results = json.load(f)
    
    # Get files that successfully type check in untyped version
    successful_files = {
        filename for filename, result in untyped_results.items() 
        if result.get('error_count', 0) == 0 and result.get('isCompiled', False)
    }
    
    print(f"Found {len(successful_files)} files with successful untyped type checking")
    
    llm_files = {
        "O3-mini": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
        "DeepSeek": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
        "Claude3-Sonnet": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",     
    }
    
    # Store all parameter and return type info for each model
    all_model_data = {}
    
    for model_name, file_path in llm_files.items():
        print(f"\nLoading {model_name}...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        model_data = {}
        
        for filename, file_info in data.items():
            if filename not in successful_files:
                continue
                
            # file_info is a dict with function names as keys
            for func_name, type_info_list in file_info.items():
                func_key = f"{filename}::{func_name}"
                
                for type_info in type_info_list:
                    if type_info.get('category') == 'arg':
                        # This is a parameter
                        param_name = type_info.get('name', 'unknown')
                        param_key = f"{func_key}::param_{param_name}"
                        type_list = type_info.get('type', [])
                        # Check if type is Any/blank (empty list or contains Any)
                        model_data[param_key] = len(type_list) == 0 or 'Any' in type_list
                    
                    elif type_info.get('category') == 'return':
                        # This is a return type
                        return_key = f"{func_key}::return"
                        type_list = type_info.get('type', [])
                        # Check if type is Any/blank (empty list or contains Any)
                        model_data[return_key] = len(type_list) == 0 or 'Any' in type_list
        
        all_model_data[model_name] = model_data
        print(f"  Collected {len(model_data)} type annotations")
    
    # Find common Any/blank types across all models
    model_names = list(all_model_data.keys())
    all_keys = set(all_model_data[model_names[0]].keys())
    for model_name in model_names[1:]:
        all_keys = all_keys.intersection(set(all_model_data[model_name].keys()))
    
    common_any_params = []
    common_any_returns = []
    
    for key in all_keys:
        # Check if this key is Any/blank in ALL models
        is_any_in_all = all(all_model_data[model_name][key] for model_name in model_names)
        
        if is_any_in_all:
            if '::param_' in key:
                common_any_params.append(key)
            elif '::return' in key:
                common_any_returns.append(key)
    
    return {
        'common_any_params': common_any_params,
        'common_any_returns': common_any_returns,
        'total_common_params': len(common_any_params),
        'total_common_returns': len(common_any_returns),
        'total_analyzed_keys': len(all_keys)
    }

if __name__ == "__main__":
    results = analyze_common_any_blank_types()
    
    print("\n" + "="*60)
    print("COMMON ANY/BLANK TYPES ACROSS ALL 3 LLMs")
    print("="*60)
    
    print(f"\nTotal type annotations analyzed: {results['total_analyzed_keys']}")
    print(f"Common Any/blank parameters: {results['total_common_params']}")
    print(f"Common Any/blank returns: {results['total_common_returns']}")
    
    if results['total_common_params'] > 0:
        print(f"\nCommon Any/blank parameters (showing first 10):")
        for param in results['common_any_params'][:10]:
            print(f"  {param}")
        if len(results['common_any_params']) > 10:
            print(f"  ... and {len(results['common_any_params']) - 10} more")
    
    if results['total_common_returns'] > 0:
        print(f"\nCommon Any/blank returns (showing first 10):")
        for ret in results['common_any_returns'][:10]:
            print(f"  {ret}")
        if len(results['common_any_returns']) > 10:
            print(f"  ... and {len(results['common_any_returns']) - 10} more")
