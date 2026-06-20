# Prompt Recipe Notes

## What is a prompt recipe?

A template for generating synthetic data. Different recipes produce different quality outputs.

## Recipes I've tested

### Alpaca style
```
Below is an instruction. Write a response.

### Instruction:
{instruction}

### Response:
```
- Works well for general Q&A
- Tends to produce verbose outputs
- Good for: explanations, definitions

### Short answer
```
Answer this question in 1-2 sentences:
{instruction}
```
- Produces concise outputs
- Good for: factual Q&A, flashcards
- Bad for: code, creative writing

### Code task
```
Write code to solve this problem:
{instruction}

Provide only the code, no explanation.
```
- Produces clean code
- Problem: model sometimes adds explanation anyway
- Need post-processing to extract just the code

### Chain of thought
```
Think step by step:
{instruction}

Show your reasoning, then give the final answer.
```
- Produces detailed reasoning
- Good for: math, logic, multi-step problems
- Bad for: simple Q&A (too verbose)

## Lessons learned

1. **Shorter prompts often work better** - long system prompts confuse small models
2. **Temperature matters a lot** - 0.7 gives variety, 0.3 gives consistency
3. **Batch size affects quality** - generating 100 at once vs 1 at a time can give different distributions
4. **Post-processing is essential** - raw LLM output needs cleanup
