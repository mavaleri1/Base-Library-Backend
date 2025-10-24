import os
import yaml
from jinja2 import Template
from typing import Dict, Any


def load_yaml_with_env(path: str) -> Dict[str, Any]:
    """
    Loads YAML file with environment variable substitution via Jinja2.
    
    Args:
        path: Path to YAML file
        
    Returns:
        dict: Loaded configuration with substituted variables
    """
    # Check if file is prompts.yaml
    if path.endswith("prompts.yaml"):
        # For prompts.yaml file, don't render Jinja2 templates
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f.read())
    else:
        # For other files, apply environment variable substitution
        with open(path, "r", encoding="utf-8") as f:
            template = Template(f.read())
            rendered = template.render(env=os.environ)
            return yaml.safe_load(rendered)