import os
import sys

# Windows terminal encoding fix
sys.stdout.reconfigure(encoding='utf-8')

# Ensure Python can find our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from alignment.inference.aligned_generator import load_aligned_model, generate_structured
except ImportError as e:
    print(f"ERROR Importing module: {e}")
    sys.exit(1)

model_path = os.path.abspath("src/alignment/models/dpo_aligned_model")

print("="*60)
print("TESTING THE NEW DPO ALIGNED CONTRACT SENSE")
print("="*60)

# Check if model exists
if not os.path.exists(model_path) or not os.listdir(model_path):
    print(f"ERROR: Could not find model files at {model_path}")
    print("Please make sure you extracted dpo_model.zip into this folder!")
    sys.exit(1)

# Load the model
try:
    print("Loading model into memory (this will take a moment)...")
    # Using 4-bit just for quick local testing to prevent OOM
    model, tokenizer = load_aligned_model(model_path, use_4bit=True)
except Exception as e:
    print(f"ERROR Loading Model: {e}")
    sys.exit(1)

# Test Clauses
test_cases = [
    {
        "clause": "The receiving party shall hold the Confidential Information in strict confidence and shall not disclose such Confidential Information to any third party without prior written consent, except where required by law, court order, or governmental regulation, provided prompt notice is given.",
        "query": "Is there any exception where they can share my data without my permission?"
    }
]

print("\n" + "="*60)
print("INFERENCE TEST")
print("="*60)

for case in test_cases:
    print(f"\n[QUERY]: {case['query']}")
    print("-" * 40)
    
    # Run the aligned generator
    result = generate_structured(model, tokenizer, case["clause"], case["query"])
    
    # Print the raw string exactly as the LLM generated it to prove formatting
    print("\n[MODEL OUTPUT]:\n")
    print(result.get("raw_output", "No output generated."))
    
    print("\n" + "-" * 40)
    print(f"Parse Success (Perfect Formatting): {'YES' if result['parse_success'] else 'NO'}")
    print("="*60)
