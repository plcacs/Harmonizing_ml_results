import json
import os

def analyze_any_blank_types():
    """Analyze LLM files to count Any/blank parameter and return types"""
    
    # Load untyped mypy results to filter successful files
    untyped_mypy_path = "../../mypy_results/mypy_outputs/mypy_results_untyped_with_errors.json"
    
    with open(untyped_mypy_path, 'r') as f:
        untyped_results = json.load(f)
    
    # Get files that successfully type check in untyped version
    successful_files = {
        filename for filename, result in untyped_results.items() 
        if result.get('error_count', 0) == 0 and result.get('isCompiled', False)
    }
    
    llm_files = {
        "O3-mini": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
        "DeepSeek": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
        "Claude3-Sonnet": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",     
    }
    
    results = {}
    
    for model_name, file_path in llm_files.items():
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        param_any_blank = 0
        return_any_blank = 0
        total_params = 0
        total_returns = 0
        analyzed_files = 0
        
        for filename, file_info in data.items():
            # Only analyze files that successfully type check in untyped version
            if filename not in successful_files:
                continue
                
            analyzed_files += 1
            
            if 'functions' in file_info:
                for func in file_info['functions']:
                    # Check parameters
                    if 'parameters' in func:
                        for param in func['parameters']:
                            total_params += 1
                            param_type = param.get('type', '').strip()
                            if param_type in ['', 'Any', 'typing.Any']:
                                param_any_blank += 1
                    
                    # Check return type
                    if 'return_type' in func:
                        total_returns += 1
                        return_type = func['return_type'].strip()
                        if return_type in ['', 'Any', 'typing.Any']:
                            return_any_blank += 1
        
        results[model_name] = {
            'param_any_blank': param_any_blank,
            'return_any_blank': return_any_blank,
            'total_params': total_params,
            'total_returns': total_returns,
            'analyzed_files': analyzed_files,
            'param_percentage': (param_any_blank / total_params * 100) if total_params > 0 else 0,
            'return_percentage': (return_any_blank / total_returns * 100) if total_returns > 0 else 0
        }
    
    return results

if __name__ == "__main__":
    results = analyze_any_blank_types()
    
    # Write results to file
    with open('any_analysis_results.txt', 'w') as f:
        f.write("Any/Blank Type Analysis Results\n")
        f.write("="*50 + "\n\n")
        
        for model, stats in results.items():
            f.write(f"{model}:\n")
            f.write(f"  Files analyzed: {stats['analyzed_files']}\n")
            f.write(f"  Parameters typed as Any/blank: {stats['param_any_blank']} ({stats['param_percentage']:.1f}%)\n")
            f.write(f"  Returns typed as Any/blank: {stats['return_any_blank']} ({stats['return_percentage']:.1f}%)\n\n")
    
    print("Analysis complete! Results saved to any_analysis_results.txt")


