# ACMA
ACMA is a multi-agent framework that enhances multi-tool task execution in large language models through multi-level experience reuse and dynamic strategy selection, improving efficiency and reducing costs.

## data
corpus_embeddings.pt

🛠️ StableToolBench
A powerful and stable toolkit for tool learning and evaluation.

🚀 Quick Start
1. Environment Setup
Create a dedicated conda environment with Python 3.10:

bash
# Create and activate conda environment
conda create -n acma python=3.10 -y
conda activate acma

## Install dependencies
pip install -r requirements.txt
2. Configuration
Before running the toolkit, you need to configure your API credentials:

Modify api_key and base_url in the configuration files according to your service provider

Ensure you have valid API access credentials

3. Running the Toolkit
Navigate to the StableToolBench directory and launch the main program:

bash
cd stabletoolbench
python main.py
4. Automation with Shell Script
For convenience, you can use the provided shell script to automate the entire process:

bash
## Make the script executable (if needed)
chmod +x run.sh

## Execute the automation script
./run.sh
The run.sh script handles:

Environment activation

Configuration validation

Main program execution

Error handling and logging

📋 Requirements
Python 3.10+

Conda package manager

Valid API credentials

📁 Project Structure
text
ACMA/
├── acma/                       # Core ACMA implementation
│   ├── [core files and directories]
│   ├── arguments.py
│   ├── main.py
│   ├── ...
│   └── utils.py
├── requirements.txt
├── stabletoolbench/            # StableToolBench integration
│   ├── config.yml
│   ├── main.py
│   └── ...
├── tools/                       # API tool categories
│   ├── Advertising/
│   ├── Business/
│   ├── Finance/
│   ├── Health_and_Fitness/
│   ├── Social/
│   ├── ... 
│   └── eCommerce/
└── transformers/                # Pre-trained models
    ├── 2025-07-24_02-49-05/
    ├── all-mpnet-base-v2/
    └── gpt2/
🔧 Troubleshooting
If you encounter any issues:

Ensure your Python version is 3.10: python --version

Verify all dependencies are installed: pip list

Check your API credentials are correctly set

Review logs in the logs/ directory
