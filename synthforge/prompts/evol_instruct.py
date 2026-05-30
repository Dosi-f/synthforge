"""
Evol-Instruct prompt templates.

Based on the WizardLM Evol-Instruct approach:
https://arxiv.org/abs/2304.12244

The core idea: start with a simple instruction and iteratively
"evolve" it to be more complex, nuanced, or constrained.
"""

EVOL_INSTRUCT_SYSTEM = """You are an expert at creating high-quality training data for language models.

Your task is to generate diverse, well-structured instruction-following examples that can be used to fine-tune LLMs.

Follow these guidelines:
1. Instructions should be clear and unambiguous
2. Responses should be thorough, accurate, and well-formatted
3. Vary the complexity — include both simple and multi-step tasks
4. Vary the domain — rotate through coding, writing, reasoning, analysis, and creative tasks
5. Include edge cases and constraints in some instructions to test robustness
6. Format each example as a valid JSON object with "instruction" and "response" fields

Generate examples that would genuinely improve a model's capabilities."""

EVOL_INSTRUCT_USER = """Generate {num_samples} diverse instruction-following examples.

Topic domain: {domain}
Complexity level: {complexity}
Additional constraints: {constraints}

Output format: valid JSON array of objects with "instruction" and "response" keys."""
