finish_function = {
    "name": "finish_func",
    "description": "If you think you have finished, call this function.",
    "parameters": {
        "type": "object",
        'properties': {
    }
    }
}

finish_func = {
    "name": "Finish",
    "description": "1. If all the instructions in the user's input question have been completed, or the return of the tool expresses the meaning that it can be solved, call this function to provide the final_answer, which should include all the results the user needs.2. If the current tools cannot complete the task, call this function to restart and give the reason. The name of the failed function or the current unsolvable problem should be mentioned.3. If the task cannot be completed not because of the lack of tools, call this function to end and give the reason.  Remember: You must ALWAYS call this function at the end of your attempt and return the return_type. The final return needs to be in dictionary type.",
    "parameters": {
        "type": "object",
        "properties": {
            "return_type": {
                "type": "string",
                "enum": ["give_answer","give_up_and_restart", "give_up"],
            },
            "final_answer": {
                "type": "string",
                "description": "The final answer you want to give the user. It should not contain any sorry message. The content should include the information obtained in the previous steps, rather than just a similar message like 'The task has been completed.' You should have this field if \"return_type\"==\"give_answer\"",
            },
            "reason": {
                "type": "string",
                "description": "The reason why you give up. You should mention the names of the failed functions. You should have this field if \"return_type\"==\"give_up\" or \"return_type\"==\"give_up_and_restart\"",
            }
        },
        "required": ["return_type"],
    }
  }

decompose_task_function = {
    'name': 'decompose_task',
    'description': (
        'Decompose a complex user query into multiple subtasks. '
        'Each subtask should represent an independent, atomic action that can be '
        'handled by a single API or tool. Used when the user query involves '
        'multiple steps or objectives.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'The original user query that needs to be decomposed.'
            }
        },
        'required': ['query']
    }
}

search_apis_function = {
    'name': 'search_apis',
    'description': (
        'Search for relevant APIs or tools that can accomplish a given subtask. '
        'It first checks the L1 and L2 cache to find matching APIs. '
        'If no match is found, it performs a vector-based semantic search in the tool library.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'subtask': {
                'type': 'string',
                'description': 'The description of the subtask for which relevant APIs need to be found.'
            },
            'sublabel': {
                'type': 'string',
                'description': 'The label representing the category or type of the subtask.'
            },
            'top_k': {
                'type': 'integer',
                'description': 'Number of most relevant APIs to return (default 5).',
                'default': 5
            }
        },
        'required': ['subtask', 'sublabel']
    }
}

choose_api_function = {
    'name': 'choose_api',
    'description': (
        'Select the most appropriate API from a candidate list returned by the search step. '
        'The model considers task relevance, efficiency, and past success statistics '
        'to make the final decision.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'candidate_apis': {
                'type': 'array',
                'description': 'A list of candidate APIs retrieved from search_apis.',
                'items': {
                    'type': 'object',
                    'properties': {
                        'api_name': {'type': 'string'},
                        'description': {'type': 'string'},
                        'category': {'type': 'string'}
                    },
                    'required': ['api_name']
                }
            },
            'subtask': {
                'type': 'string',
                'description': 'The subtask description for which the API is being chosen.'
            }
        },
        'required': ['candidate_apis', 'subtask']
    }
}

check_task_completion_function = {
    'name': 'check_task_completion',
    'description': (
        'Check whether the currently selected APIs can completely solve the user query. '
        'If any parts of the task remain unsatisfied, generate a list of unmet subtasks '
        'that require further decomposition or new API search.'
    ),
    "parameters": {
        "type": "object",
        'properties': {
    }
    }
}

delete_api_function = {
    'name': 'delete_api',
    'description': (
        'Deletes a specific API from the tool pool. '
        'This function requires the API name or its unique identifier to locate and remove the API. '
        'It is used when an API is no longer needed or should be removed from the list of available tools.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'api_name': {
                'type': 'string',
                'description': 'The name of the API that needs to be deleted from the tool pool.'
            },
            'api_id': {
                'type': 'string',
                'description': 'The unique identifier of the API to delete (optional, either api_name or api_id should be provided).'
            }
        },
        'required': ['api_name'] 
    }
}
