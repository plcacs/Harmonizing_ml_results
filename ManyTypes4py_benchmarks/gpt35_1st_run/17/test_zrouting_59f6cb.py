from typing import Dict, List, Any, Union

class RoutingTestCase(unittest.TestCase):

    def setUp(self) -> None:
        ...

    def tearDown(self) -> None:
        ...

    def test_config(self) -> None:
        ...

    def test_config_precedence(self) -> None:
        ...

    def test_routing(self) -> None:
        ...

class DummyConfigPlugin(unittest.TestCase, PluginBase):

    def pre_receive(self, alert: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        ...

    def post_receive(self, alert: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        ...

    def status_change(self, alert: Dict[str, Any], status: str, text: str, **kwargs: Any) -> Tuple[Dict[str, Any], str, str]:
        ...

class DummyPagerDutyPlugin(PluginBase):

    def pre_receive(self, alert: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        ...

    def post_receive(self, alert: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        ...

    def status_change(self, alert: Dict[str, Any], status: str, text: str, **kwargs: Any) -> Tuple[Dict[str, Any], str, str]:
        ...

class DummySlackPlugin(PluginBase):

    def pre_receive(self, alert: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        ...

    def post_receive(self, alert: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        ...

    def status_change(self, alert: Dict[str, Any], status: str, text: str, **kwargs: Any) -> Tuple[Dict[str, Any], str, str]:
        ...

def rules(alert: Dict[str, Any], plugins: Dict[str, Any], **kwargs: Any) -> Tuple[List[Any], Dict[str, Any]]:
    ...
