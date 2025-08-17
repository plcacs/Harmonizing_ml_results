import os
import json
import subprocess
import sys
from pathlib import Path

def run_lizard_analysis(directory_path):
    """Run lizard analysis on all Python files in a directory and return the output"""
    try:
        # Find all Python files in the directory
        python_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        if not python_files:
            print(f"No Python files found in {directory_path}")
            return None
        
        print(f"Found {len(python_files)} Python files to analyze...")
        
        # Run lizard on each Python file individually and combine results
        all_output = []
        successful_files = 0
        
        for i, python_file in enumerate(python_files, 1):
            if i % 50 == 0:  # Progress report every 50 files
                print(f"Processed {i}/{len(python_files)} files...")
            
            try:
                result = subprocess.run(
                    ['lizard', python_file],
                    capture_output=True,
                    text=True,
                    timeout=30  # Add timeout to prevent hanging
                )
                if result.returncode == 0:
                    all_output.append(result.stdout)
                    successful_files += 1
                else:
                    # Don't print error for every file, just count them
                    pass
            except subprocess.TimeoutExpired:
                print(f"Timeout analyzing {python_file}")
                continue
            except Exception as e:
                # Don't print error for every file, just count them
                continue
        
        print(f"Successfully analyzed {successful_files}/{len(python_files)} files")
        
        if not all_output:
            print("No files were successfully analyzed")
            return None
            
        return '\n'.join(all_output)
    except FileNotFoundError:
        print("Lizard not found. Please install it with: pip install lizard")
        return None

def parse_lizard_output(output):
    """Parse lizard output to extract file and function information"""
    files_data = {}
    current_file = None
    
    lines = output.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and headers
        if not line or line.startswith('=') or line.startswith('NLOC') or line.startswith('--'):
            continue
            
        # Check if this is a file summary line (contains file path)
        if '@' in line and ('ManyTypes4py_benchmarks' in line or '.py' in line):
            # Extract file path from the location part
            parts = line.split('@')
            if len(parts) >= 3:
                file_path = parts[2]
                if file_path not in files_data:
                    files_data[file_path] = {
                        'functions': [],
                        'summary': {}
                    }
                current_file = file_path
                
                # Parse function data
                try:
                    # Format: NLOC CCN token PARAM length location
                    data_parts = line.split()
                    if len(data_parts) >= 6:
                        function_data = {
                            'nloc': int(data_parts[0]),
                            'ccn': int(data_parts[1]),
                            'token': int(data_parts[2]),
                            'param': int(data_parts[3]),
                            'length': int(data_parts[4]),
                            'function_name': parts[0].split('@')[0] if '@' in parts[0] else parts[0]
                        }
                        files_data[current_file]['functions'].append(function_data)
                except (ValueError, IndexError):
                    continue
                    
        # Check if this is a file summary line (at the end)
        elif 'file analyzed' in line or 'files analyzed' in line:
            # Look for the summary table above this line
            for i in range(len(lines) - 1, -1, -1):
                summary_line = lines[i].strip()
                if summary_line and not summary_line.startswith('=') and not summary_line.startswith('NLOC'):
                    try:
                        parts = summary_line.split()
                        if len(parts) >= 6 and current_file is not None:
                            files_data[current_file]['summary'] = {
                                'total_nloc': int(parts[0]),
                                'avg_nloc': float(parts[1]),
                                'avg_ccn': float(parts[2]),
                                'avg_token': float(parts[3]),
                                'function_count': int(parts[4]),
                                'file_path': parts[5] if len(parts) > 5 else current_file
                            }
                            break
                    except (ValueError, IndexError):
                        continue
    
    return files_data

def analyze_complexity(directory_path):
    """Main function to analyze complexity and create JSON output"""
    # Get directory name for output file
    dir_name = os.path.basename(directory_path.rstrip('/\\'))
    
    # Create output directory
    output_dir = "complexity_of_source_codes"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{dir_name}_complexity_analysis.json")
    
    print(f"Analyzing complexity for directory: {directory_path}")
    
    # Run lizard analysis
    lizard_output = run_lizard_analysis(directory_path)
    if not lizard_output:
        return
    
    # Parse the output
    files_data = parse_lizard_output(lizard_output)
    
    # Process the data to create the required format
    result = {}
    
    for file_path, data in files_data.items():
        if not data['functions']:
            continue
            
        # Calculate metrics
        functions = data['functions']
        avg_ccn = sum(f['ccn'] for f in functions) / len(functions)
        
        # Get top-3 functions by CCN
        top_3_functions = sorted(functions, key=lambda x: x['ccn'], reverse=True)[:3]
        top_3_ccn = [f['ccn'] for f in top_3_functions]
        
        # Get total line count
        total_lines = sum(f['nloc'] for f in functions)
        
        # Check if CCN is high
        is_ccn_high = avg_ccn > 10
        
                # Create result for this file
        file_name = os.path.basename(file_path)
        result[file_name] = {
            'average_CCN': round(avg_ccn, 3),
            'top_3_functions_CCN': top_3_ccn,
            'total_line_count': total_lines,
            'function_count': len(functions)
        }
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis complete! Results saved to: {output_file}")
    print(f"Analyzed {len(result)} files")

if __name__ == "__main__":
   
    #directory_paths = ["original_files" "claude3_sonnet_1st_run","gpt35_2nd_run","gpt4o","o1_mini","o3_mini_1st_run"]
    directory_paths = ["claude3_sonnet_1st_run"]
    for directory_path in directory_paths:
        analyze_complexity(directory_path)
