#!/usr/bin/env python3
"""
Simple script to run the type hint extraction analysis.
"""

from extract_type_hints import process_typing_analysis

if __name__ == "__main__":
    input_file = "file_typing_analysis_deepseek.json"
    output_file = "detailed_type_hint_analysis.json"

    print("Starting type hint analysis...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    try:
        process_typing_analysis(input_file, output_file)
        print("Analysis completed successfully!")
    except Exception as e:
        print(f"Error during analysis: {e}")
