#!/usr/bin/env bash
# SynthForge Setup Script
# Sets up the development environment with GPU support detection.

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== SynthForge Setup ===${NC}"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "Python version: ${python_version}"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo -e "\n${GREEN}Creating virtual environment...${NC}"
    python3 -m venv .venv
else
    echo -e "\n${YELLOW}Virtual environment already exists${NC}"
fi

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install base dependencies
echo -e "\n${GREEN}Installing base dependencies...${NC}"
pip install -e ".[dev]"

# GPU detection
echo -e "\n${GREEN}Checking GPU availability...${NC}"

if command -v rocm-smi &> /dev/null; then
    gpu_info=$(rocm-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")
    if [ -n "$gpu_info" ]; then
        echo -e "GPU detected:"
        echo "$gpu_info" | while read line; do
            echo -e "  ${GREEN}✓${NC} $line"
        done
        echo -e "\n${GREEN}Installing GPU dependencies...${NC}"
        pip install -e ".[gpu]"
    else
        echo -e "${YELLOW}rocm-smi found but no usable GPU detected${NC}"
    fi
else
    echo -e "${YELLOW}No AMD GPU detected — skipping GPU dependencies${NC}"
    echo -e "${YELLOW}Cloud API backends (OpenAI, Anthropic) will still work${NC}"
fi

# Create directories
mkdir -p outputs .cache logs

# Copy .env if it doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "\n${YELLOW}Created .env from .env.example — please edit with your API keys${NC}"
fi

# Run basic test
echo -e "\n${GREEN}Running quick health check...${NC}"
python -c "from synthforge import Generator; print('Import OK')" || {
    echo -e "${RED}Import check failed — check your installation${NC}"
    exit 1
}

echo -e "\n${GREEN}=== Setup complete! ===${NC}"
echo -e "Next steps:"
echo -e "  1. Edit .env with your API keys"
echo -e "  2. Run: python examples/basic_generation.py"
echo -e "  3. Or:  synthforge --help"
