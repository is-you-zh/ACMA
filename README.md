# ACMA
ACMA is a multi-agent framework that enhances multi-tool task execution in large language models through multi-level experience reuse and dynamic strategy selection, improving efficiency and reducing costs.

## data
corpus_embeddings.pt


# ToolCombine Setup and Usage

## Create a New Environment
To get started, first create a new Python environment using Conda:

```bash
conda create -n toolcombine python=3.10
conda activate toolcombine```

Then, install the required dependencies:

pip install -r requirements.txt
API Configuration

Modify the api key and base url to match your setup.

Navigate to the StableToolBench Directory:
cd StableToolBench-master
Run the Tool:

To start the tool, simply run the following:

python main.py
Running via run.sh

For ease, you can use the provided run.sh script to set up and run everything automatically. To execute the script, simply run:

bash run.sh
🛠️ StableToolBench
A powerful and stable toolkit for tool learning and evaluation.

🚀 Quick Start
1. Environment Setup
Create a dedicated conda environment with Python 3.10:

bash
# Create and activate conda environment
conda create -n toolcombine python=3.10 -y
conda activate toolcombine

# Install dependencies
pip install -r requirements.txt
2. Configuration
Before running the toolkit, you need to configure your API credentials:

Modify api_key and base_url in the configuration files according to your service provider

Ensure you have valid API access credentials

3. Running the Toolkit
Navigate to the StableToolBench directory and launch the main program:

bash
cd StableToolBench-master
python main.py
4. Automation with Shell Script
For convenience, you can use the provided shell script to automate the entire process:

bash
# Make the script executable (if needed)
chmod +x run.sh

# Execute the automation script
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
StableToolBench-master/
├── main.py              # Main entry point
├── requirements.txt     # Python dependencies
├── run.sh              # Automation script
└── config/             # Configuration files
    ├── api_config.yaml # API settings
    └── ...
🔧 Troubleshooting
If you encounter any issues:

Ensure your Python version is 3.10: python --version

Verify all dependencies are installed: pip list

Check your API credentials are correctly set

Review logs in the logs/ directory
