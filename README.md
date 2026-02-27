# 🚀 Fine-Tuning Qwen2.5-3B-Instruct using GRPO & PEFT

This repository demonstrates how to fine-tune **Qwen/Qwen2.5-3B-Instruct** using **GRPO (Generalized Reward Policy Optimization)** combined with **PEFT (LoRA)** for efficient and reward-based training. The objective is to improve task-specific performance while reducing computational cost by updating only a small subset of model parameters instead of the entire model.

## 📌 Overview

- **Base Model:** Qwen/Qwen2.5-3B-Instruct  
- **Training Method:** GRPO (Reinforcement Learning-based optimization)  
- **Fine-Tuning Technique:** LoRA (Parameter-Efficient Fine-Tuning)  
- **Frameworks Used:** PyTorch, Transformers, TRL, PEFT  
- **Goal:** Memory-efficient and scalable fine-tuning  

## 📂 Repository Structure

```
.
├── Fine-tuning-qwen2-5-3b-instruct-GRPO-peft.ipynb   # Main training notebook  
├── README.md                                         # Project documentation  
└── (Optional folders for configs, data, scripts)  
```

## ⚙️ Installation

Ensure Python 3.9+ is installed.

```bash
pip install torch transformers datasets trl peft accelerate bitsandbytes
```

(Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

## 🚀 Usage

Clone the repository:

```bash
git clone https://github.com/siddhartth04/Fine-tuning-qwen2-5-3b-instruct-GRPO-peft.git
cd Fine-tuning-qwen2-5-3b-instruct-GRPO-peft
```

Launch the notebook:

```bash
jupyter notebook Fine-tuning-qwen2-5-3b-instruct-GRPO-peft.ipynb
```

The notebook covers:

- Loading the base model
- Applying LoRA adapters
- Configuring GRPO trainer
- Training loop
- Evaluation and saving adapters

## 🧠 Training Workflow

1. Load base model from Hugging Face
2. Apply LoRA using PEFT
3. Define reward function for GRPO
4. Train with GRPO trainer
5. Evaluate results
6. Save fine-tuned adapters

## 📄 Example: LoRA Configuration

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_id = "Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

## 🖥️ Hardware Requirements

- GPU recommended (≥16GB VRAM preferred)
- Supports mixed precision (fp16/bf16)
- Compatible with bitsandbytes for memory-efficient training

## 📜 License

Refer to the LICENSE file for licensing details.

---

⭐ If you find this project useful, consider giving it a star.
