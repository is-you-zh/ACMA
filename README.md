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


### Key Points:
- Ensure that each block of code is wrapped in triple backticks (```) followed by the language identifier (`bash` or `python`) to ensure proper syntax highlighting.
- The `README` sections are now properly formatted, and everything should appear as intended on GitHub.

Let me know if you need further assistance!
