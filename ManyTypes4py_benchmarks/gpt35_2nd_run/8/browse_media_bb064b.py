from typing import List, Dict, Any, Optional, Union

def _item_to_children_media_class(item: Dict[str, Any], info: Optional[Dict[str, Any]] = None) -> MediaClass:
def _item_to_media_class(item: Dict[str, Any], parent_item: Optional[Dict[str, Any]] = None) -> MediaClass:
def _list_payload(item: Dict[str, Any], children: Optional[List[Dict[str, Any]]] = None) -> BrowseMedia:
def _raw_item_payload(entity: Any, item: Dict[str, Any], parent_item: Optional[Dict[str, Any]] = None, title: Optional[str] = None, info: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, MediaClass, MediaType, bool]]:
def _item_payload(entity: Any, item: Dict[str, Any], parent_item: Dict[str, Any]) -> BrowseMedia
async def browse_top_level(media_library: Any) -> BrowseMedia:
async def browse_node(entity: Any, media_library: Any, media_content_type: str, media_content_id: str) -> BrowseMedia:
