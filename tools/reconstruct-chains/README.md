# Reconstruct Chains

This directory is split by framework, with shared utilities in `common/`.

## Layout

- `common/`
  - `lsp_client.py`: shared JSON-RPC LSP client
  - `render_report.py`: shared markdown/mermaid renderer
- `torch_npu/`
  - `run.py`: torch_npu chain reconstruction
  - `README.md`: torch_npu usage and schema
- `mindspore/`
  - `run.py`: scaffold entrypoint (not implemented yet)
  - `README.md`
