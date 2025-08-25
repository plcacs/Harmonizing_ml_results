from typing import List, Optional, Dict, Any, Tuple

def process_data(data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    result = []
    error = None
    
    try:
        for item in data:
            processed_item = {}
            
            if "name" in item:
                processed_item["name"] = item["name"].upper()
            
            if "age" in item:
                processed_item["age"] = item["age"] * 2
            
            if "tags" in item and isinstance(item["tags"], list):
                processed_item["tags"] = [tag.lower() for tag in item["tags"]]
            
            result.append(processed_item)
    except Exception as e:
        error = str(e)
    
    return result, error
