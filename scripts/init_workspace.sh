#!/bin/bash
# Initialize workspace symlinks for local development
# GitHub shows submodule links, but locally we use symlinks for flexibility

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$SCRIPT_DIR/../workspace"

# Default paths (modify these to match your local setup)
# Example: /Users/fanzhilan/project/mindspore-agent/workspace-mindspore-framework/code/
DEFAULT_MINDSPORE_PATH="/path/to/mindspore"
DEFAULT_OP_PLUGIN_PATH="/path/to/op-plugin"
DEFAULT_ACLNN_DASHBOARD_PATH="/path/to/aclnn-dashboard"

# Use environment variables if set, otherwise use defaults
MINDSPORE_PATH="${MINDSPORE_PATH:-$DEFAULT_MINDSPORE_PATH}"
OP_PLUGIN_PATH="${OP_PLUGIN_PATH:-$DEFAULT_OP_PLUGIN_PATH}"
ACLNNDASHBOARD_PATH="${ACLNNDASHBOARD_PATH:-$DEFAULT_ACLNN_DASHBOARD_PATH}"

echo "Initializing workspace symlinks..."
echo ""

# Function to create symlink
create_symlink() {
    local src="$1"
    local dst="$2"
    local name="$3"
    
    if [ -L "$dst" ]; then
        echo "[$name] Removing existing symlink: $dst"
        rm "$dst"
    elif [ -e "$dst" ]; then
        echo "[$name] Warning: $dst exists and is not a symlink. Skipping."
        return 1
    fi
    
    if [ -d "$src" ]; then
        ln -sf "$src" "$dst"
        echo "[$name] Created symlink: $dst -> $src"
        return 0
    else
        echo "[$name] Warning: Source path does not exist: $src"
        echo "[$name] Please clone the repository or update the path."
        return 1
    fi
}

# Create symlinks
create_symlink "$MINDSPORE_PATH" "$WORKSPACE_DIR/mindspore" "mindspore"
create_symlink "$OP_PLUGIN_PATH" "$WORKSPACE_DIR/op-plugin" "op-plugin"
create_symlink "$ACLNNDASHBOARD_PATH" "$WORKSPACE_DIR/aclnn-dashboard" "aclnn-dashboard"

echo ""
echo "Workspace initialization complete!"
echo ""
echo "To customize paths, set environment variables before running:"
echo "  MINDSPORE_PATH=/your/path/mindspore \\"
echo "  OP_PLUGIN_PATH=/your/path/op-plugin \\"
echo "  ACLNN_DASHBOARD_PATH=/your/path/aclnn-dashboard \\"
echo "  ./scripts/init_workspace.sh"
