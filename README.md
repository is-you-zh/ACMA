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
```

### 2. API Configuration

Before launching the framework, you'll need to configure your API credentials:

- Locate the configuration files in the `stabletoolbench/` directory
- Update `api_key` and `base_url` parameters with your service provider credentials
- Verify that your API keys have the necessary permissions for tool access

### 3. Launching the Framework

Navigate to the StableToolBench integration module and start the main program:

```bash
cd stabletoolbench
python main.py
```

### 4. Automated Execution
For streamlined operation, we provide an automation script that handles the entire workflow:

```bash
# Grant execution permissions (first time only)
chmod +x run.sh

# Execute the complete pipeline
./run.sh
```


## 📋 System Requirements

- **Python**: Version 3.10 or higher
- **Package Manager**: Conda (recommended for environment isolation)
- **API Access**: Valid credentials for tool execution
- **Storage**: Sufficient space for embeddings and model checkpoints
- **Memory**: 16GB+ RAM recommended for optimal performance

## 📁 Repository Structure
ACMA/
├── acma/ # Core framework implementation
│ ├── arguments.py # CLI argument parsing
│ ├── main.py # Entry point for ACMA
│ ├── utils.py # Utility functions
│ ├── models.py # Neural model definitions
│ ├── tool_task.py # Tool task management
│ ├── retrieval.py # Experience retrieval mechanisms
│ ├── [additional core modules] # Supporting implementation files
│ └── ...
│
├── stabletoolbench/ # Tool learning and evaluation toolkit
│ ├── config.yml # API and system configuration
│ ├── main.py # StableToolBench entry point
│ └── ...
│
├── tools/ # Comprehensive API tool collection
│ ├── Advertising/
│ ├── Business/
│ ├── Finance/
│ ├── Health_and_Fitness/
│ ├── Social/
│ ├── ... (additional categories)
│ └── eCommerce/
│
├── transformers/ # Pre-trained language models
│ ├── all-mpnet-base-v2/ # Sentence embedding model
│ ├── gpt2/ # Base language model
│ └── [timestamped checkpoints]/ # Training artifacts
│
└── requirements.txt # Python dependencies
