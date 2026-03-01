from config import *
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken
import json
from config import *
from arguments import parse_args
from termcolor import colored
import threading
from openai import OpenAI

MAX_TOKENS = 4096
enc = tiktoken.encoding_for_model("gpt-4")
args = parse_args()
output_dir = args.output_dir
client = OpenAI(
        api_key=api_key,
        base_url=base_url   
    )

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def gpt(prompt, model="gpt-4.1", temperature=1.0, max_tokens=MAX_TOKENS, n=1, stop=None, tools=None, max_retries=5, timeout=30):
    messages = [{"role": "user", "content": prompt}]
    outputs = []
    total_tokens_used = 0

    def call_gpt():
        nonlocal res, error
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                n=cnt,
                functions=tools,
            )
        except Exception as e:
            error = e

    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            res, error = None, None
            thread = threading.Thread(target=call_gpt)
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                print(f"Timeout occurred, retrying {attempt}/{max_retries}...")
                continue

            if error:
                if attempt == max_retries:
                    raise RuntimeError("GPT request failed after multiple retries.")
                print(f"Error occurred: {error}, retrying {attempt}/{max_retries}...")
                continue

            for choice in res.choices:
                if choice.finish_reason == 'function_call':
                    content = choice.message.content.strip() if choice.message.content else None
                    func_args = choice.message.function_call.arguments
                    try:
                        parsed_args = json.loads(func_args)
                        args_str = json.dumps(parsed_args, ensure_ascii=False)
                    except json.JSONDecodeError:
                        args_str = func_args  
                    combined = (content or "") + args_str
                    outputs.append(combined)
                else:
                    outputs.append(choice.message.content)

            if hasattr(res, "usage") and res.usage is not None:
                total_tokens_used += res.usage.total_tokens
                print(f"[OpenAI] Tokens used this call: {res.usage.total_tokens}")
            else:
                prompt_tokens = len(enc.encode(str(messages)))
                output_tokens = sum(len(enc.encode(choice.message.content)) for choice in res.choices)
                token_cnt = prompt_tokens + output_tokens
                total_tokens_used += token_cnt
                print(f"[Fallback] Tokens estimated: {token_cnt}")

            break

    return outputs, total_tokens_used

def gpt_no_tools(prompt, model="gpt-4.1", temperature=1.0, max_tokens=MAX_TOKENS, n=1, stop=None, tools=None, max_retries=5, timeout=20):
    messages = [{"role": "user", "content": prompt}]
    outputs = []
    total_tokens_used = 0

    def call_gpt():
        nonlocal res, error
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=cnt,
                stop=stop,
            )
        except Exception as e:
            error = e

    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            res, error = None, None
            thread = threading.Thread(target=call_gpt)
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                print(f"Timeout occurred, retrying {attempt}/{max_retries}...")
                continue

            if error:
                if attempt == max_retries:
                    raise RuntimeError("GPT request failed after multiple retries.")
                print(f"Error occurred: {error}, retrying {attempt}/{max_retries}...")
                continue

            for choice in res.choices:
                outputs.append(choice.message.content)

            # Token usage tracking
            if hasattr(res, "usage") and res.usage is not None:
                total_tokens_used += res.usage.total_tokens
                # print(f"[OpenAI] Tokens used this value: {res.usage.total_tokens}")
            else:
                prompt_tokens = len(enc.encode(str(messages)))
                output_tokens = sum(len(enc.encode(choice.message.content)) for choice in res.choices)
                token_cnt = prompt_tokens + output_tokens
                total_tokens_used += token_cnt
                print(f"[Fallback] Tokens estimated: {token_cnt}")

            break

    return outputs, total_tokens_used

def call_gpt(messages, model, functions=None):
    messages_converted = messages
    @retry(wait=wait_random_exponential(multiplier=10, max=50), stop=stop_after_attempt(5))
    def call_gpt_retry(messages, model, functions):
        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = client.chat.completions.create(
                            seed=123,
                            messages=messages,
                            functions=functions,
                            model=model,
                            # max_tokens = 1024, 
                        )
            except Exception as e:
                exception[0] = e
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=30)  

        if thread.is_alive():
            print("Function timed out, retrying...")
            raise TimeoutError("Function execution exceeded the timeout limit")
        if exception[0] is not None:
            raise exception[0]

        return result[0]

    try:
        response = call_gpt_retry(messages_converted, model, functions)
        if response.choices[0].finish_reason == 'function_call':
            response_json = json.loads(response.json())
            tool_call = {'arguments': response_json['choices'][0]['message']['function_call']['arguments'], 'name': response_json['choices'][0]['message']['function_call']['name']}
            response.choices[0].message.tool_calls = [dotdict({'id':'111', 'function':dotdict(tool_call)})]
        if response.usage is None:
            token_cnt = len(enc.encode(str(functions))) + len(enc.encode(str(messages))) + len(enc.encode(str(response.choices[0].message.content)))
            response.usage = dotdict({'total_tokens': token_cnt})
        else:
            print(colored('tokens', 'blue'), colored(response.usage.total_tokens, 'blue'))
        return response
        
    except Exception as e:
        raise e

def call_gpt_no_func(messages,model):
    @retry(wait=wait_random_exponential(multiplier=60, max=100), stop=stop_after_attempt(10))
    def call_gpt_retry(messages):
        result = [None]
        exception = [None]
        def target():
            try:
                result[0] = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout = 30)  
        if thread.is_alive():
            print("Function timed out, retrying...")
            raise TimeoutError("Function execution exceeded the timeout limit")
        if exception[0] is not None:
            raise exception[0]
        return result[0]

    return call_gpt_retry(messages)

def gpt_for_data(prompt,model=model_name,temperature=1.0):
    messages = [{"role": "user", "content": prompt}]
    try:
        res = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature
        )
        return res
    except Exception as e:
        print(e)

def gpt_exp(messages, model=model_name, temperature=0.1):
    try:
        res = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
        )
        return res
    except Exception as e:
        print(e)

