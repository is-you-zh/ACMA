from prompt import *
from config import *
from models import gpt
from transformers import GPT2Tokenizer
import random

tokenizer = GPT2Tokenizer.from_pretrained("../../transformers/gpt2")
max_token_length = 4096

def get_token_length(text):
    return len(tokenizer.encode(text))

class ToolTask():
    """
    Input (x)   : a text instruction
    Output (y)  : a text generation
    Reward (r)  : # TODO
    Input Example: 
    Output Example: 
    """
    def __init__(self):
        """
        file: a text file, each line is some sentences
        """
        super().__init__()
        self.steps = 7
        self.stops = ['\nObservation:\n', None]
        self.value_cache = {}
    
    @staticmethod
    def generate_self_reflection(unique_trajectories, question, tool_des, functions):
        reflection_mapping = []
        trajectories = ""

        sampled_items = random.sample(unique_trajectories, min(3, len(unique_trajectories)))
        failed_trajectories = "\n".join([f"{question}\n{traj}\n" for traj in unique_trajectories])

        failed_trajectories = [f"Question: {traj}" for traj in failed_trajectories.split("Question: ")[1:]]
        
        for traj in failed_trajectories:
            trajectories += traj
            
            reflect_prompt = reflection_prompt.format(tool_des=tool_des, question=question, trajectory=traj, functions=functions)
            
            reflection = gpt(reflect_prompt, model=model_name)
            
            trajectories += "Reflection: " + reflection[0] + "\n"
            
            reflection_mapping.append({
                'question': question,
                'trajectory': traj,
                'reflection': reflection[0]
            })

        return reflection_mapping

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = '', reflection_mapping_list=[], tool_des=None, functions=None):
        question = x
        input = x + y
        trajectories = ""
        if reflection_mapping_list:
            for reflection_mapping in reflection_mapping_list:
                traj_with_reflection = reflection_mapping['trajectory'] + "FAILED TRAJECTORY\nReflection: " + reflection_mapping['reflection'] + "\n\n"
                trajectories += traj_with_reflection
            
            prompt = cot_prompt_feedback.format(trajectories=trajectories, input=input, tool_des=tool_des)
            if get_token_length(prompt) > max_token_length:
                print("Too long")
                trajectories = ""
                for reflection_mapping in reflection_mapping_list[:3]:
                    traj_with_reflection = reflection_mapping['trajectory'] + "FAILED TRAJECTORY\nReflection: " + reflection_mapping['reflection'] + "\n\n"
                    trajectories += traj_with_reflection
                prompt = cot_prompt_feedback_short.format(trajectories=trajectories, input=input, tool_des=tool_des)
            
            return prompt
        else:
            prompt = cot_prompt.format(input=input, functions=functions, tool_des=tool_des)
            if get_token_length(prompt) > max_token_length:
                prompt = cot_prompt_short.format(input=input, tool_des=tool_des)
            return prompt
  
    @staticmethod
    def cot_prompt_wrap_with_tool(x: str, y: str = '', reflection_mapping_list=[], tool_des=None, functions=None):
        question = x
        input = x +"\n"+ y
        # input = x
        trajectories = ""
        if reflection_mapping_list:
            for reflection_mapping in reflection_mapping_list:
                traj_with_reflection = reflection_mapping['trajectory'] + "FAILED TRAJECTORY\nReflection: " + reflection_mapping['reflection'] + "\n\n"
                trajectories += traj_with_reflection
            
            prompt = cot_prompt_feedback.format(trajectories=trajectories, input=input, tool_des=tool_des)
            if get_token_length(prompt) > max_token_length:
                print("Too long")
                trajectories = ""
                for reflection_mapping in reflection_mapping_list[:3]:
                    traj_with_reflection = reflection_mapping['trajectory'] + "FAILED TRAJECTORY\nReflection: " + reflection_mapping['reflection'] + "\n\n"
                    trajectories += traj_with_reflection
                prompt = cot_prompt_feedback_short.format(trajectories=trajectories, input=input, tool_des=tool_des)
            
            return prompt
        else:
            prompt = cot_prompt_with_tool.format(input=input, tool_des=tool_des)
            if get_token_length(prompt) > max_token_length:
                prompt = cot_prompt_short.format(input=input, tool_des=tool_des)
            return prompt

    @staticmethod
    def value_prompt_wrap(x: str, y: str, z: list = [], reflections: list = []) -> str:
        inp = y + "\nThus the correctess score is "
        prompt = value_prompt_reasoning.format(s="", input=inp)
            
        return prompt

    @staticmethod
    def value_prompt_wrap_together(x: str, y: str, z: list = [], reflections: list = []) -> str:
        inp = y + "\nThus the correctess score is "
        prompt = value_prompt_reasoning_together.format(s="", input=inp)
            
        return prompt

    @staticmethod
    def value_outputs_unwrap(evaluate_prompt: str):
        score = evaluate_prompt[0]
        if '10' in score:
            return 1.0
        elif '9' in score:
            return 0.9
        elif '8' in score:
            return 0.8
        elif '7' in score:
            return 0.7
        elif '6' in score:
            return 0.6
        elif '5' in score:
            return 0.5
        elif '4' in score:
            return 0.4
        elif '3' in score:
            return 0.3
        elif '2' in score:
            return 0.2
        elif '1' in score:
            return 0.1
        else:
            return -1
            
    @staticmethod
    def value_outputs_unwrap_together(evaluate_prompt: list):
        scaled_list = [round(x * 0.1, 2) for x in evaluate_prompt]
        return scaled_list