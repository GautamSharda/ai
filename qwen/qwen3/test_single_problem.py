"""
Interactive testing script for trying different prompts on a single ARC-AGI problem.
Edit the PROMPT variable to test different approaches.
"""
import json
import time
import sys
import os

# Add the qwen3 directory to the path
sys.path.insert(0, '/ai_network_volume/ai/qwen/qwen3')
os.chdir('/ai_network_volume/ai/qwen/qwen3')

# Import using importlib since filename has dots
import importlib.util
spec = importlib.util.spec_from_file_location("qwen3", "/ai_network_volume/ai/qwen/qwen3/qwen3_0.6b.py")
qwen3_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen3_module)
qwen3_tg = qwen3_module.qwen3_tg

# Load the ARC-AGI 2025 evaluation set
eval_challenges_path = '/ai_network_volume/ai/arc-agi/arc-agi-2025/arc-agi_evaluation_challenges.json'
eval_solutions_path = '/ai_network_volume/ai/arc-agi/arc-agi-2025/arc-agi_evaluation_solutions.json'

with open(eval_challenges_path, 'r') as f:
    eval_challenges = json.load(f)

with open(eval_solutions_path, 'r') as f:
    eval_solutions = json.load(f)

# Get first problem
problem_id = list(eval_challenges.keys())[0]
problem = eval_challenges[problem_id]
solution = eval_solutions[problem_id]

# ============================================================================
# EDIT THIS PROMPT TO TEST DIFFERENT APPROACHES
# ============================================================================
MAX_NEW_TOKENS = 20  # Adjust as needed

def create_prompt(problem):
    """
    Create your prompt here. The problem dict has:
    - 'train': list of {'input': grid, 'output': grid}
    - 'test': list of {'input': grid} (no output provided)
    """
    train_examples = problem.get('train', [])
    test_input = problem.get('test', [{}])[0].get('input', [])

    # Build prompt with all train examples
    prompt_parts = ["You are given training examples showing input-output pairs. Learn the pattern and predict the output for the test input.\n"]

    # Add all training examples
    for i, example in enumerate(train_examples, 1):
        prompt_parts.append(f"\nTraining Example {i}:")
        prompt_parts.append(f"Input: {example['input']}")
        prompt_parts.append(f"Output: {example['output']}")

    # Add test input
    prompt_parts.append(f"\nTest Input:")
    prompt_parts.append(f"{test_input}")
    prompt_parts.append(f"\nPredict the output:")

    prompt = "\n".join(prompt_parts)
    return prompt

# ============================================================================

def format_grid(grid):
    """Pretty print a grid"""
    if not grid:
        return "[]"
    rows = []
    for row in grid[:5]:  # Show first 5 rows
        rows.append(str(row[:10]))  # Show first 10 cols
    result = "\n".join(rows)
    if len(grid) > 5:
        result += f"\n... ({len(grid)} total rows)"
    return result

print("=" * 80)
print(f"PROBLEM: {problem_id}")
print("=" * 80)

# Show train examples
print("\nTRAIN EXAMPLES:")
for i, example in enumerate(problem.get('train', [])[:2], 1):  # Show first 2
    print(f"\nExample {i}:")
    print(f"  Input shape: {len(example['input'])}x{len(example['input'][0]) if example['input'] else 0}")
    print(f"  Output shape: {len(example['output'])}x{len(example['output'][0]) if example['output'] else 0}")
    print(f"  Input preview:\n{format_grid(example['input'])}")
    print(f"  Output preview:\n{format_grid(example['output'])}")

# Show test input
test_input = problem.get('test', [{}])[0].get('input', [])
print(f"\nTEST INPUT:")
print(f"  Shape: {len(test_input)}x{len(test_input[0]) if test_input else 0}")
print(f"  Preview:\n{format_grid(test_input)}")

# Show expected output
expected_outputs = solution
print(f"\nEXPECTED OUTPUT(S): {len(expected_outputs)} possible solution(s)")
for i, expected in enumerate(expected_outputs[:1], 1):  # Show first expected output
    print(f"  Solution {i} shape: {len(expected)}x{len(expected[0]) if expected else 0}")
    print(f"  Preview:\n{format_grid(expected)}")

# Generate prompt
prompt = create_prompt(problem)
print("\n" + "=" * 80)
print("PROMPT:")
print("=" * 80)
print(prompt)
print("\n" + "=" * 80)
print("RUNNING INFERENCE...")
print("=" * 80)

# Measure inference
start_time = time.time()
try:
    response = qwen3_tg(prompt, max_new_tokens=MAX_NEW_TOKENS)
    elapsed = time.time() - start_time
    tokens_per_sec = MAX_NEW_TOKENS / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 80)
    print("RESPONSE:")
    print("=" * 80)
    print(response)

    print("\n" + "=" * 80)
    print("METRICS:")
    print("=" * 80)
    print(f"Time: {elapsed:.2f}s")
    print(f"Tokens/sec: {tokens_per_sec:.2f}")
    print(f"Max tokens: {MAX_NEW_TOKENS}")

    # Try to check if correct (very basic check)
    print("\n" + "=" * 80)
    print("CORRECTNESS CHECK:")
    print("=" * 80)
    # This is a simple check - real ARC-AGI requires parsing the output grid
    correct = "unknown"
    print(f"Correct: {correct} (requires manual inspection)")

except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n✗ ERROR after {elapsed:.2f}s:")
    print(f"  {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
