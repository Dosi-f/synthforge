"""
Persona-driven generation prompt templates.

Generates synthetic conversations where the assistant adopts
a specific persona/style/role.
"""

PERSONA_SYSTEM = """You are simulating conversations for training data generation.

You will be given a persona description and a scenario. Generate realistic
multi-turn conversations where the assistant consistently embodies the persona.

The conversation should feel natural — users may ask follow-up questions,
challenge the assistant, or change topics slightly."""

PERSONA_USER = """Persona: {persona_description}

Scenario: {scenario}

Number of turns: {num_turns}

Generate a natural conversation between a user and an assistant embodying this persona.
Output as a JSON array of message objects with "role" and "content" fields.

Make the assistant responses distinctive to the persona — vary vocabulary,
tone, level of detail, and communication style."""
