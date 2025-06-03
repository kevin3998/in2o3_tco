"""
Module: general_utils
Functionality: Contains general utility classes and functions used across the application.
               Currently includes the PromptManager class for managing and formatting
               prompt templates for different domains and languages.
"""
from typing import Dict, Tuple

class PromptManager:
    def __init__(self):
        self.templates: Dict[Tuple[str, str], str] = {}

    def add_prompt(self, domain: str, language: str, template: str):
        self.templates[(domain, language)] = template

    def get_prompt(self, domain: str, language: str, text_to_format: str) -> str:
        template = self.templates.get((domain, language))
        if not template:
            raise ValueError(f"Prompt template not found for domain '{domain}' and language '{language}'")
        # Ensure the template is formatted with the provided text
        # The original code had template.format(text=text), which is good.
        # If 'text' is always the key, this is fine.
        # For more flexibility, you might pass a dict: template.format(**format_args)
        return template.format(text=text_to_format)