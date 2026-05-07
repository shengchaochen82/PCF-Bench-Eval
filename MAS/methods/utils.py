import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def handle_retry_error(retry_state):
    exc = retry_state.outcome.exception()
    print(f"Error in calling LLM: {exc}")
    raise exc