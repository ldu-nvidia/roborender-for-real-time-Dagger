"""The trusted Image3F pickle protocol must never bind a network interface."""
from test.deployment.source_paths import PIPELINE_ROOT
import ast
from pathlib import Path
from unittest.mock import Mock

import pytest


def test_image3f_listener_binds_only_loopback():
    source = PIPELINE_ROOT / 'dagger/image3f_server.py'
    tree = ast.parse(source.read_text())
    listener = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'tcp_listener')
    module = ast.fix_missing_locations(ast.Module(body=[listener], type_ignores=[]))
    server = Mock()
    server.accept.side_effect = StopIteration('one bind is enough')
    sockets = Mock()
    sockets.socket.return_value = server
    scope = {'socket': sockets, 'print': Mock(), 'threading': Mock()}
    exec(compile(module, str(source), 'exec'), scope)
    with pytest.raises(StopIteration):
        scope['tcp_listener'](object(), object(), 18500)
    server.bind.assert_called_once_with(('127.0.0.1', 18500))
    server.listen.assert_called_once_with(8)


def test_renderer_execution_evidence_is_bounded_and_keeps_genuine_counts():
    source = PIPELINE_ROOT / 'dagger/image3f_server.py'
    tree = ast.parse(source.read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'execution_evidence')
    module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    scope = {}
    exec(compile(module, str(source), 'exec'), scope)
    result = scope['execution_evidence']({'request_index': 2, 'trt_invocations': 4,
        'tea_cache_block_stack_calls': 4, 'finite': True, 'private_internal': 'excluded'})
    assert result['request_index'] == 2
    assert result['trt_invocations'] == result['tea_cache_block_stack_calls'] == 4
    assert result['finite'] is True
    assert 'private_internal' not in result
    assert scope['execution_evidence'](None) == {}
