class test_LiveCheckSensor:
    @pytest.fixture()
    def sensor(self) -> LiveCheckSensor:
        return LiveCheckSensor()

    def test_on_stream_event__no_test(self, *, sensor: LiveCheckSensor):
        stream = Mock()
        stream.current_test = None
        event = Mock()
        event.headers = {}
        state = sensor.on_stream_event_in(('topic', 'foo'), 3, stream, event)
        sensor.on_stream_event_out(('topic', 'foo'), 3, stream, event, state)

    def test_on_stream_event(self, *, sensor: LiveCheckSensor, execution: TestExecution):
        stream = Mock()
        stream.current_test = None
        event = Mock()
        event.headers = execution.as_headers()
        assert current_test_stack.top is None
        state = sensor.on_stream_event_in(('topic', 'foo'), 3, stream, event)
        assert current_test_stack.top.id == execution.id
        assert stream.current_test.id == execution.id
        sensor.on_stream_event_out(('topic', 'foo'), 3, stream, event, state)
        assert current_test_stack.top is None
        assert stream.current_test is None

class test_LiveCheck:
    @pytest.mark.parametrize('kwarg,value,expected_value', [('test_topic_name', 'test-topic', 'test-topic'), ('bus_topic_name', 'bus-topic', 'bus-topic'), ('report_topic_name', 'report-topic', 'report-topic'), ('bus_concurrency', 1000, 1000), ('test_concurrency', 1001, 1001), ('send_reports', False, False)])
    def test_constructor(self, kwarg: str, value: str, expected_value: str, app: LiveCheck):
        app = LiveCheck('foo', **{kwarg: value})
        assert getattr(app, kwarg) == value

    # ... (rest of the code)
