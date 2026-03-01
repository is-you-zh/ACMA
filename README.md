# ACMA: Adaptive Collaborative Multi-Agent Framework

ACMA is a sophisticated multi-agent framework designed to enhance multi-tool task execution in large language models. By implementing multi-level experience reuse and dynamic strategy selection mechanisms, ACMA significantly improves execution efficiency while reducing computational costs. The framework enables intelligent orchestration of diverse tools, allowing agents to learn from past experiences and adapt their strategies in real-time.

## 📊 Data Assets

The framework comes with pre-computed embeddings to accelerate tool retrieval and selection:

- `corpus_embeddings.pt`: Pre-trained embeddings for the tool corpus, enabling efficient semantic search and tool matching

## 🛠️ StableToolBench Integration

StableToolBench serves as the foundational toolkit for tool learning and evaluation within the ACMA ecosystem. It provides a robust and stable environment for testing and benchmarking multi-tool interactions.

## 🚀 Getting Started

### 1. Environment Configuration

Begin by setting up a dedicated Python environment with all necessary dependencies:

```bash
# Create and activate a fresh conda environment
conda create -n acma python=3.10 -y
conda activate acma

# Install required packages
pip install -r requirements.txt
