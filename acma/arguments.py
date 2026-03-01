import argparse

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, choices=['gpt-5.1-chat', 'gpt-4', 'gpt-4o-mini'], default='gpt-4o-mini')
    args.add_argument('--temperature', type=float, default=1.0)
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=10)
    args.add_argument('--n_generate_sample', type=int, default=5)  
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--iterations', type=int, default=8)
    args.add_argument('--log_path', type=str)
    args.add_argument("--query_path", type=str, default='./query_dataset/anytoolbench.json', help="Path to the query data")
    args.add_argument("--output_dir", type=str, default='./', help="Directory for the output file")
    args.add_argument("--leaf_tool_number", type=int, default=5, help="Maximum number of leaf tools")
    args.add_argument("--max_api_number", type=int, default=32, help="Maximum number of API calls")

    args.add_argument("--check_solvable", action='store_true', default=False, help="check solvable")
    args.add_argument("--recheck_solved", action='store_true', default=False, help="check solvable")
    args.add_argument("--include_unsolvable", action='store_true', default=False, help="whether skip unsolvable")
    args.add_argument("--use_original_prompt", action='store_true', default=False, help="whether use original prompt")
    args.add_argument("--score_path", default=False, help="record the score")

    args = args.parse_args()
    return args
