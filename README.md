# Op-Agent

## Structure

```
op-agent/
├── .codex/ or .xxx                  # AI IDE initialization config (if needed..)
├── agents/                          # Agent definitions and Prompt templates
│   ├── prompts/                     # prompts templates
│   │   └── ....yaml                 
│   └── schemas/                     # Agent input/output Schema
│       └── ...                      # e.g. tool_call.json           
├── skills/                          # Skills callable by Agents (Tools wrapper layer)
│   └── ...
├── tools/                           # Low-level tool implementations (no Agent logic)
│   ├── reconstruct_chains/          # lsp based top-op -> aclnn chain reconstruction
│   │   └── ...
│   └── ...
├── workspace/                       # External repositories (GitHub links + local symlinks)
│   ├── aclnn-dashboard/             # ACLNN visualization platform → local symlink
│   ├── op-plugin/                   # Operator plugin repository → local symlink
│   └── mindspore/                   # MindSpore source repository → local symlink
├── runtime/                         # Agent execution engine
│   ├── ...
│   └── hooks.py                     # Lifecycle hooks (e.g., pre-process dependency check)
├── docs/                            # Knowledge base (RAG/reference) 
│   ├── knowledge_base/
│   └── strategies/                  
│       └── torch_npu_chain_reconstruction_strategy.md
├── scripts/
│   ├── init_workspace.sh            # Initialize workspace symlinks
│   └── ...                          # e.g. setup_dev.sh
└── README.md
```

## Draft Design Notes

### 1. Tools Layer (`tools/`)
Pure low-level tools without Agent business logic, can run independently.
- `tools/reconstruct_chains/`: LSP-based API sinking chain reconstruction tool
  - Entry: `tools/reconstruct_chains/torch_npu/run.py`
  - Output: Chain call graphs (forward/backward)
- **Rule**: Tools accept explicit parameters, return structured data (JSON/chart paths), do not process natural language.

### 2. Skills Layer (`skills/`)
Skill units callable by Agents, adapter layer for Tools.

### 3. Agents Layer (`agents/`)
Prompt templates and Agent configurations, **no code logic**.
- `prompts/*.yaml`: Contains System Prompt and Tool descriptions
- ...

### 4. Workspace Layer (`workspace/`)
External repository dependencies.

**Design**: GitHub displays clickable submodule links, but locally use symlinks for flexibility.

| Repository | GitHub Link | Local Setup |
|------------|-------------|-------------|
| `mindspore/` | [Gitcode](https://gitcode.com/mindspore/mindspore) | `ln -sf /your/path/mindspore workspace/mindspore` |
| `op-plugin/` | [Gitcode](https://gitcode.com/Ascend/op-plugin) | `ln -sf /your/path/op-plugin workspace/op-plugin` |
| `aclnn-dashboard/` | [GitHub](https://github.com/Fzilan/aclnn-dashboard) | `ln -sf /your/path/aclnn-dashboard workspace/aclnn-dashboard` |

**Notes**
- Each developer may have different local paths for these large repositories
- Avoid redundant clones (reuse existing local repos)
- Submodules are configured with `update = none` (won't auto-clone)

**Setup script** (`scripts/init_workspace.sh`):
```bash
#!/bin/bash
# Configure your local paths
MINDSPORE_PATH="/path/to/mindspore"
OP_PLUGIN_PATH="/path/to/op-plugin"
ACLNNDASHBOARD_PATH="/path/to/aclnn-dashboard"

# Create symlinks
ln -sf "$MINDSPORE_PATH" workspace/mindspore
ln -sf "$OP_PLUGIN_PATH" workspace/op-plugin
ln -sf "$ACLNNDASHBOARD_PATH" workspace/aclnn-dashboard

echo "Workspace initialized."
```