# ACMA
ACMA is a multi-agent framework that enhances multi-tool task execution in large language models through multi-level experience reuse and dynamic strategy selection, improving efficiency and reducing costs.

## data
corpus_embeddings.pt


# ToolCombine Setup and Usage

## Create a New Environment
To get started, first create a new Python environment using Conda:

```bash
conda create -n toolcombine python=3.10
conda activate toolcombine

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

This script will automatically handle the environment setup, API configuration, and execution steps for you.


### Explanation:
- **Environment Setup**: The first part details creating the Conda environment, activating it, and installing dependencies.
- **API Configuration**: This section instructs the user to modify the API key and base URL, followed by navigating to the `StableToolBench-master` directory and running the tool.
- **`run.sh`**: Provides a convenient way to automate the process, with a simple command to execute the script.

You can now paste this directly into your `README` on GitHub!
