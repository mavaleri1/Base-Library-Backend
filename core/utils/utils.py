import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any
from jinja2 import Template


class Config:
    """Class for loading and managing configuration"""

    def __init__(self):
        self.prompts_config_path = os.getenv(
            "PROMPTS_CONFIG_PATH", "./configs/prompts.yaml"
        )
        self.graph_config_path = os.getenv("GRAPH_CONFIG_PATH", "./configs/graph.yaml")
        self.main_dir = os.getenv("MAIN_DIR", "./data")

    def load_prompts(self) -> Dict[str, str]:
        """Loads prompts from YAML file"""
        with open(self.prompts_config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_graph_config(self) -> Dict[str, Any]:
        """Loads graph configuration from YAML file"""
        with open(self.graph_config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_model_name(self) -> str:
        """Returns model name from graph.yaml"""
        graph_conf = self.load_graph_config()
        return graph_conf.get("model_config", {}).get("name", "gpt-4.1-mini")

    def ensure_directories(self):
        """Creates necessary directories"""
        Path(self.main_dir).mkdir(exist_ok=True)
        Path(f"{self.main_dir}/outputs").mkdir(exist_ok=True)


def pretty_print_pydantic(pydantic_model) -> str:
    """Nicely formats JSON schema of Pydantic model"""
    return json.dumps(pydantic_model.model_json_schema(), indent=4)


def render_system_prompt(
    template_type: str, template_variant: str = "initial", **kwargs: Any
) -> str:
    """
    Renders system prompt based on template type and variant.

    Args:
        template_type: Template type (e.g., 'generating_content')
        template_variant: Template variant ('initial' or 'further')
        **kwargs: Parameters for template substitution

    Returns:
        Rendered prompt
    """
    config = Config()
    prompts_config = config.load_prompts()

    # Form key for template search
    if template_variant == "initial":
        template_key = f"{template_type}_system_prompt"
    else:
        template_key = f"{template_type}_{template_variant}_system_prompt"

    # If specific variant not found, use base
    if template_key not in prompts_config:
        template_key = f"{template_type}_system_prompt"

    if template_key not in prompts_config:
        raise KeyError(f"Template '{template_key}' not found in prompts config")

    template_content = prompts_config[template_key]
    template = Template(template_content)

    return template.render(**kwargs)
