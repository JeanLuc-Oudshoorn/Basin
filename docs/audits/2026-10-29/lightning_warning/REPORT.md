# Lightning's pkg_resources notice

The vendored RWKV training backend imports the pinned Lightning 1.9.5 runtime.
Its Fabric namespace registration calls deprecated `pkg_resources`, producing
a UserWarning during backend import. This notice does not mean that training,
forecasting, model loading or the Hugging Face cache failed.

The backend wrapper now filters only the `pkg_resources is deprecated as an API`
UserWarning originating in `lightning_fabric`, and only while importing the
vendored model. Other warnings and import/runtime errors remain visible. The
warning filter is scoped and restored immediately afterward. No model or
dependency version was changed, and running processes need not be interrupted;
future processes use the filter.

Verification used a mocked backend import to emit both the known Fabric notice
and an unrelated warning. Only the unrelated warning remained visible, and the
original working directory was restored after import. Ruff checks passed.
