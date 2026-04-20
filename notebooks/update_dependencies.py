import json
from pathlib import Path

notebook_path = Path(r"c:\Users\Jay\Desktop\DAU\SEM-2\DL\Project\ContractSense-copilot\notebooks\06_dpo_alignment_colab.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Unpin strict versions to fix Triton / CUDA / Bitsandbytes compat issues with new Colab PyTorch
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if '!pip install' in source_str:
            cell['source'] = [
                "# Install dependencies using flexible versions to adapt to Colab's changing PyTorch/Triton environment.\n",
                "# This fixes the 'triton.ops' and bitsandbytes CUDA binary errors.\n",
                "!pip install -q -U transformers trl peft accelerate bitsandbytes datasets matplotlib seaborn pandas\n"
            ]
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print(f"Updated {notebook_path} with flexible dependencies")
