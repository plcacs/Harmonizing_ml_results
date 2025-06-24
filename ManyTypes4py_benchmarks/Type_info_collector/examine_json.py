import json

def examine_json_structure(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        print(f"\n=== {filename} ===")
        print(f"Type: {type(data)}")
        print(f"Keys count: {len(data)}")
        
        # Sample first few keys and their values
        sample_keys = list(data.keys())[:3]
        print(f"Sample keys: {sample_keys}")
        
        for key in sample_keys:
            value = data[key]
            print(f"\nKey: {key}")
            print(f"Value type: {type(value)}")
            if isinstance(value, dict):
                print(f"Dict keys: {list(value.keys())[:5]}")
                # Show first dict value
                first_val = list(value.values())[0] if value else None
                print(f"First value type: {type(first_val)}")
                if isinstance(first_val, list):
                    print(f"List length: {len(first_val)}")
                    if first_val:
                        print(f"First list item: {first_val[0]}")
            elif isinstance(value, list):
                print(f"List length: {len(value)}")
                if value:
                    print(f"First list item: {value[0]}")
            else:
                print(f"Value: {value}")
                
    except Exception as e:
        print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    examine_json_structure('Type_info_original_files.json')
    examine_json_structure('Type_info_deep_seek_benchmarks.json') 