#!/bin/bash
# Build script for Agent Studio WASM

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔨 Building Agent Studio WASM..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack not found. Install with: cargo install wasm-pack"
    exit 1
fi

# Build the WASM package
echo "📦 Running wasm-pack build..."
wasm-pack build --target web --out-dir pkg --out-name rust_ai_agents_studio

# Create dist directory
mkdir -p dist

# Copy files to dist
echo "📁 Copying files to dist..."
cp index.html dist/
cp -r pkg dist/

echo "✅ Build complete!"
echo "📂 Output: $SCRIPT_DIR/dist/"
echo ""
echo "To serve locally:"
echo "  cd dist && python3 -m http.server 8080"
echo ""
echo "Or copy dist/ contents to dashboard/static/ for integration"
