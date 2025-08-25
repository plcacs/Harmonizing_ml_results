from typing import List, Optional, Dict, Any, Tuple

def process_data(data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    result = []
    error = None
    
    try:
        for item in data:
            if "name" in item and "value" in item:
                processed_item: Dict[str, Any] = {
                    "name": item["name"],
                    "processed_value": item["value"] * 2 if isinstance(item["value"], (int, float)) else item["value"]
                }
                result.append(processed_item)
    except Exception as e:
        error = str(e)
    
    return result, error
