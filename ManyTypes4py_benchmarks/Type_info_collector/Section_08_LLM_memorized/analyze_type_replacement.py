import json
import csv
from collections import defaultdict


def load_type_info(file_path):
    """Load type information from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_function_mappings(mappings_file):
    """Load function name mappings from JSON file."""
    try:
        with open(mappings_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading function mappings: {e}")
        return None


def is_concrete_type(type_str):
    """Check if type is concrete (not Any, None, or empty)."""
    if not type_str or not isinstance(type_str, str):
        return False
    type_str = type_str.strip().lower()
    return type_str not in ["any", "none", ""]


def analyze_type_replacements(original_data, renamed_data, function_mappings):
    """Analyze how renamed versions replace original LLM concrete types with Any."""
    replacements = 0
    total_concrete = 0
    preserved = 0
    debug_info = []

    # Find common files
    common_files = set(original_data.keys()) & set(renamed_data.keys())
    print(f"Found {len(common_files)} common files")
    
    for filename in common_files:
        original_functions = original_data[filename]
        renamed_functions = renamed_data[filename]
        
        # Get function mappings for this file
        file_mappings = function_mappings.get(filename, {})
        if not file_mappings:
            print(f"  Skipping {filename} - no mappings found")
            continue

        print(f"  Processing {filename} with {len(file_mappings)} function mappings")

        for original_func_name, renamed_func_name in file_mappings.items():
            # Check if both functions exist in any scope
            original_func_found = False
            renamed_func_found = False
            original_func_key = None
            renamed_func_key = None
            
            # Look for functions in any scope (global, class, etc.)
            for func_key in original_functions.keys():
                # Extract base function name by removing scope suffix (e.g., "@global@" or "@ClassName@")
                if '@' in func_key:
                    base_name = func_key.split('@')[0]
                else:
                    base_name = func_key
                
                if base_name == original_func_name:
                    original_func_key = func_key
                    original_func_found = True
                    break
            
            for func_key in renamed_functions.keys():
                # Extract base function name by removing scope suffix
                if '@' in func_key:
                    base_name = func_key.split('@')[0]
                else:
                    base_name = func_key
                
                if base_name == renamed_func_name:
                    renamed_func_key = func_key
                    renamed_func_found = True
                    break
            
            # Check if both functions exist
            if not original_func_found or not renamed_func_found:
                print(f"    Skipping {original_func_name} -> {renamed_func_name} - function not found in both datasets")
                continue

            original_params = original_functions[original_func_key]
            renamed_params = renamed_functions[renamed_func_key]

            print(f"    Comparing {original_func_name} -> {renamed_func_name}: {len(original_params)} vs {len(renamed_params)} params")

            # Match parameters by category and name
            original_dict = {
                (p.get("category", ""), p.get("name", "")): p for p in original_params
            }
            renamed_dict = {
                (p.get("category", ""), p.get("name", "")): p for p in renamed_params
            }

            common_keys = set(original_dict.keys()) & set(renamed_dict.keys())
            
            # Debug: show parameter keys
            if len(common_keys) == 0 and total_concrete < 3:
                print(f"      Original param keys: {list(original_dict.keys())[:5]}")
                print(f"      Renamed param keys: {list(renamed_dict.keys())[:5]}")
                print(f"      Original params sample: {original_params[:2] if original_params else 'None'}")
                print(f"      Renamed params sample: {renamed_params[:2] if renamed_params else 'None'}")

            print(f"      Found {len(common_keys)} common parameters")

            for key in common_keys:
                original_param = original_dict[key]
                renamed_param = renamed_dict[key]

                original_type = (
                    original_param.get("type", [""])[0] if original_param.get("type") else ""
                )
                renamed_type = (
                    renamed_param.get("type", [""])[0] if renamed_param.get("type") else ""
                )

                # Debug: print some type examples
                if total_concrete < 5:  # Only print first few for debugging
                    debug_info.append(f"      {key}: '{original_type}' -> '{renamed_type}'")

                # Check if original had concrete type
                if is_concrete_type(original_type):
                    total_concrete += 1

                    # Check if renamed replaced with Any
                    if renamed_type.strip().lower() == "any":
                        replacements += 1
                    else:
                        preserved += 1

    # Print debug info
    print("\nDebug - Sample type comparisons:")
    for info in debug_info[:10]:  # Show first 10
        print(info)

    replacement_rate = replacements / total_concrete if total_concrete > 0 else 0
    return replacements, total_concrete, replacement_rate


def main():
    # Load function mappings
    function_mappings = load_function_mappings("function_mappings.json")
    if not function_mappings:
        print("Failed to load function mappings")
        return

    # Define model pairs for comparison
    model_pairs = {
        "deepseek": {
            "original": "../Type_info_LLMS/Type_info_deep_seek_benchmarks.json",
            "renamed": "../Type_info_LLMS/Type_info_deepseek_renamed_output_2_benchmarks.json"
        },
        "o3-mini": {
            "original": "../Type_info_LLMS/Type_info_o3_mini_1st_run_benchmarks.json",
            "renamed": "../Type_info_LLMS/Type_info_o3_mini_renamed_output_benchmarks.json"
        },
        "claude3-sonnet": {
            "original": "../Type_info_LLMS/Type_info_claude3_sonnet_1st_run_benchmarks.json",
            "renamed": "../Type_info_LLMS/Type_info_claude_sonnet_renamed_output_benchmarks.json"
        },
        "gpt35": {
            "original": "../Type_info_LLMS/Type_info_gpt35_1st_run_benchmarks.json",
            "renamed": "../Type_info_LLMS/Type_info_gpt35_renamed_output_benchmarks.json"
        }
    }

    print("=" * 60)
    print("TYPE REPLACEMENT ANALYSIS: Original LLM → Renamed LLM")
    print("=" * 60)

    results = []

    for model_name, file_paths in model_pairs.items():
        print(f"\nProcessing {model_name}...")

        original_data = load_type_info(file_paths["original"])
        renamed_data = load_type_info(file_paths["renamed"])
        
        if not original_data:
            print(f"  Failed to load original: {file_paths['original']}")
            continue
        if not renamed_data:
            print(f"  Failed to load renamed: {file_paths['renamed']}")
            continue

        replacements, total_concrete, replacement_rate = analyze_type_replacements(
            original_data, renamed_data, function_mappings
        )

        results.append(
            {
                "Model": model_name,
                "Replacements": replacements,
                "Total_Concrete": total_concrete,
                "Replacement_Rate": replacement_rate,
                "Preserved": total_concrete - replacements,
            }
        )

        print(
            f"  Replacements: {replacements:,} / {total_concrete:,} = {replacement_rate:.3f} ({replacement_rate*100:.1f}%)"
        )

    # Save to CSV
    output_file = "type_replacement_results.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Replacements",
                "Total_Concrete",
                "Replacement_Rate_%",
                "Preserved",
            ]
        )

        for result in results:
            writer.writerow(
                [
                    result["Model"],
                    result["Replacements"],
                    result["Total_Concrete"],
                    f"{result['Replacement_Rate']*100:.2f}",
                    result["Preserved"],
                ]
            )

    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'Replacements':<12} {'%':<8}")
    print("-" * 40)
    for result in results:
        print(
            f"{result['Model']:<15} {result['Replacements']:<12,} {result['Replacement_Rate']*100:<8.1f}%"
        )


if __name__ == "__main__":
    main()
