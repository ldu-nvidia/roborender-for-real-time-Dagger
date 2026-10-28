# Test-only source

All first-party unit, regression and functional tests, fixtures and test-only
helpers live under this single repository-root `test/` directory:

```text
test/
  run_cpu_checks.sh                      CPU-only regression entrypoint
  repository/                           infrastructure + repository boundaries
  pipeline/                             BC / rendering / DAgger / evaluation
  deployment/                           deployment + recovery tests
    rtx_actor/                          Isaac adapter / reset / resource tests
    h200/                               H200 launch and transport tests
    connectivity/                       authenticated route tests
    macos/                              historical bridge tests
    throughput/                         timing-analysis tests
  rtx/                                  RTX package tests + live conformance tools
```

The pinned `pipeline/vendor` reference is unmodified and excluded from the
first-party layout rule, as are materialized external dependencies. Test
fixtures/import bootstraps stay here. Production source must not import test
modules, pytest or unittest. `repository/test_test_layout.py` enforces this
boundary. The root `pytest.ini` is the only first-party pytest configuration.

From the repository root:

```bash
ROBORENDER_TEST_PYTHON=runtime/test-venv-system/bin/python \
  bash test/run_cpu_checks.sh -q
runtime/test-venv-system/bin/python -m pytest -q test/deployment
```

The test runner disables CUDA and also checks shell syntax and whitespace.
The default CPU suite does not allocate compute, start Isaac or certify
production GPU capacity. The separately invoked `rtx/live_*_conformance.py`
tools are not collected by pytest and require their own authorized live setup.
Source-bound live qualification runners and their runtime resource/identity
guards remain in deployment source where production depends on them. They are
not unit-test modules; running them requires their separate explicit admission.
Historical documentation/receipts may retain old test paths. Those records
are not rewritten to pretend that a moved test executed under an old release.
