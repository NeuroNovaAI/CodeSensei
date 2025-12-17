
# 🤖 CodeSensei - Complete Build Guide

> An AI-powered code review assistant that provides intelligent feedback on pull requests. Built 100% free using open-source tools.

**🎯 Project Goal:** Build a production-grade ML system that demonstrates all skills required for LLM Engineer roles.

**💰 Total Cost:** $0 (completely free)

**⏱️ Timeline:** 12 weeks (15-20 hours/week)

**🔗 What You'll Build:**
- Fine-tuned LLM for code review
- Production deployment on HuggingFace Spaces
- RAG system for enhanced accuracy
- Monitoring dashboard with W&B
- GitHub bot integration
- Professional portfolio piece

---

## 📚 Table of Contents

- [Prerequisites](#prerequisites)
- [Tech Stack](#tech-stack)
- [Phase 1: Foundation (Weeks 1-4)](#phase-1-foundation-weeks-1-4)
- [Phase 2: Deployment (Weeks 5-8)](#phase-2-deployment-weeks-5-8)
- [Phase 3: Advanced Features (Weeks 9-12)](#phase-3-advanced-features-weeks-9-12)
- [Project Structure](#project-structure)
- [Skills Demonstrated](#skills-demonstrated)
- [Portfolio Impact](#portfolio-impact)

---

## Prerequisites

### Required Accounts (All Free)
- [ ] Google Account (for Colab)
- [ ] GitHub Account
- [ ] HuggingFace Account
- [ ] Weights & Biases Account

### Required Knowledge
- Basic Python programming
- Git fundamentals
- Basic understanding of ML concepts (optional but helpful)

### Hardware Requirements
- Computer with internet connection
- Web browser
- **No GPU required** (we use free Colab)

---

## Tech Stack

### Development
```
- IDE: VS Code (Free)
- Notebooks: Google Colab (Free T4 GPU)
- Version Control: Git + GitHub
- Storage: Google Drive (15GB free)
```

### ML Framework
```
- PyTorch
- HuggingFace Transformers
- HuggingFace PEFT (for LoRA/QLoRA)
- HuggingFace Datasets
- LangChain
```

### Deployment
```
- Hosting: HuggingFace Spaces
- UI: Gradio
- Model Storage: HuggingFace Hub
- Vector DB: ChromaDB (in-memory)
```

### Monitoring
```
- Experiment Tracking: Weights & Biases
- CI/CD: GitHub Actions
- Documentation: GitHub Pages
```

---

## Phase 1: Foundation (Weeks 1-4)

### 🎯 Goals
- Collect and process training data
- Fine-tune a code review model
- Understand the ML pipeline end-to-end

### 📊 Deliverables
- [ ] Dataset of 10,000+ code reviews
- [ ] Fine-tuned CodeLlama-7B model
- [ ] Evaluation metrics and benchmarks
- [ ] W&B experiment dashboard

---

### Week 1: Environment Setup & Data Collection

#### Day 1-2: Project Setup (4 hours)

**Tasks:**
1. Create GitHub repository
2. Set up project structure
3. Configure development environment
4. Create documentation template

**Step-by-Step:**

```bash
# 1. Create new repository on GitHub
# Name: codereview-ai
# Description: AI-powered code review assistant
# Initialize with README

# 2. Clone locally
git clone https://github.com/YOUR_USERNAME/codereview-ai.git
cd codereview-ai

# 3. Create project structure
mkdir -p {src,notebooks,data,models,configs,docs}
touch src/__init__.py
touch requirements.txt
touch .gitignore
```

**Create `.gitignore`:**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/

# Jupyter
.ipynb_checkpoints/

# Data
data/raw/
data/processed/
*.csv
*.json

# Models
models/*.bin
models/*.pt
*.pth

# Secrets
.env
config.yaml
```

**Create `requirements.txt`:**
```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
peft>=0.6.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
langchain>=0.1.0
chromadb>=0.4.0
gradio>=4.0.0
wandb>=0.16.0
requests>=2.31.0
beautifulsoup4>=4.12.0
python-dotenv>=1.0.0
```

**Deliverable Checklist:**
- [ ] GitHub repo created
- [ ] Project structure set up
- [ ] Dependencies documented
- [ ] README.md drafted

---

#### Day 3-5: Data Collection Pipeline (6 hours)

**Goal:** Scrape 10,000+ code reviews from GitHub

**Create `notebooks/01_data_collection.ipynb` in Colab:**

```python
# Install dependencies
!pip install requests beautifulsoup4 datasets

# Import libraries
import requests
import json
import time
from datetime import datetime
import pandas as pd
from datasets import Dataset

# GitHub API setup
GITHUB_TOKEN = "your_github_token"  # Get from github.com/settings/tokens
headers = {"Authorization": f"token {GITHUB_TOKEN}"}

def fetch_pull_requests(repo, max_prs=100):
    """Fetch PRs from a repository"""
    url = f"https://api.github.com/repos/{repo}/pulls"
    params = {"state": "closed", "per_page": 100}
    
    response = requests.get(url, headers=headers, params=params)
    return response.json()

def fetch_review_comments(repo, pr_number):
    """Fetch review comments for a PR"""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments"
    response = requests.get(url, headers=headers)
    return response.json()

# Target repositories (popular Python projects)
repos = [
    "django/django",
    "pallets/flask",
    "psf/requests",
    "pytorch/pytorch",
    "huggingface/transformers",
]

# Collect data
all_reviews = []

for repo in repos:
    print(f"Processing {repo}...")
    prs = fetch_pull_requests(repo)
    
    for pr in prs[:20]:  # Limit per repo
        pr_number = pr["number"]
        comments = fetch_review_comments(repo, pr_number)
        
        for comment in comments:
            review_data = {
                "repo": repo,
                "pr_number": pr_number,
                "code_diff": comment.get("diff_hunk", ""),
                "comment": comment.get("body", ""),
                "file_path": comment.get("path", ""),
                "language": "python"  # We're focusing on Python
            }
            all_reviews.append(review_data)
        
        time.sleep(1)  # Rate limiting

# Convert to DataFrame
df = pd.DataFrame(all_reviews)
print(f"Collected {len(df)} reviews")

# Save locally
df.to_csv("code_reviews.csv", index=False)

# Upload to HuggingFace Hub
dataset = Dataset.from_pandas(df)
dataset.push_to_hub("YOUR_USERNAME/code-review-dataset")
```

**Deliverable Checklist:**
- [ ] Data collection script working
- [ ] 10,000+ reviews collected
- [ ] Dataset uploaded to HuggingFace
- [ ] Data quality checked

---

#### Day 6-7: Data Processing (4 hours)

**Create `notebooks/02_data_processing.ipynb`:**

```python
# Load dataset
from datasets import load_dataset
import re

dataset = load_dataset("YOUR_USERNAME/code-review-dataset")

def clean_diff(diff):
    """Clean code diff"""
    # Remove git metadata
    diff = re.sub(r'^@@.*@@', '', diff, flags=re.MULTILINE)
    # Remove +/- symbols
    diff = re.sub(r'^[+\-]', '', diff, flags=re.MULTILINE)
    return diff.strip()

def create_training_example(example):
    """Format as instruction-following example"""
    code = clean_diff(example["code_diff"])
    review = example["comment"]
    
    prompt = f"""Review the following code change:

```python
{code}
```

Provide a constructive code review:"""
    
    return {
        "prompt": prompt,
        "completion": review,
        "language": example["language"]
    }

# Process dataset
processed = dataset.map(create_training_example)

# Filter out bad examples
processed = processed.filter(lambda x: len(x["completion"]) > 20)
processed = processed.filter(lambda x: len(x["prompt"]) < 2000)

# Train/validation split
split = processed.train_test_split(test_size=0.1, seed=42)

print(f"Training examples: {len(split['train'])}")
print(f"Validation examples: {len(split['test'])}")

# Save
split.push_to_hub("YOUR_USERNAME/code-review-processed")
```

**Deliverable Checklist:**
- [ ] Data cleaned and formatted
- [ ] Train/val split created
- [ ] Processed dataset uploaded
- [ ] Sample examples verified

---

### Week 2-3: Model Fine-tuning

#### Day 8-14: Fine-tune with QLoRA (12 hours)

**Create `notebooks/03_fine_tuning.ipynb` in Colab:**

```python
# Install dependencies
!pip install -q transformers peft bitsandbytes accelerate wandb

# Login to services
import wandb
from huggingface_hub import login

wandb.login()  # Enter your W&B API key
login()  # Enter your HF token

# Import libraries
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# Load processed dataset
dataset = load_dataset("YOUR_USERNAME/code-review-processed")

# Model configuration
model_name = "codellama/CodeLlama-7b-hf"

# Quantization config (fit in free Colab)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Prepare for training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize dataset
def tokenize_function(examples):
    # Combine prompt and completion
    texts = [p + " " + c for p, c in zip(examples["prompt"], examples["completion"])]
    return tokenizer(texts, truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500,
    warmup_steps=100,
    report_to="wandb",
    run_name="codereview-lora"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Train!
trainer.train()

# Save model
model.save_pretrained("codereview-7b-lora")
tokenizer.save_pretrained("codereview-7b-lora")

# Push to HuggingFace
model.push_to_hub("YOUR_USERNAME/codereview-7b-lora")
tokenizer.push_to_hub("YOUR_USERNAME/codereview-7b-lora")
```

**Training Tips:**
- Run in multiple 12-hour Colab sessions
- Save checkpoints frequently
- Monitor W&B dashboard
- Adjust batch size if OOM errors

**Deliverable Checklist:**
- [ ] Model fine-tuned successfully
- [ ] Training curves in W&B
- [ ] Model uploaded to HuggingFace
- [ ] Inference tested on examples

---

### Week 4: Evaluation & Benchmarking

#### Day 15-21: Model Evaluation (8 hours)

**Create `notebooks/04_evaluation.ipynb`:**

```python
from transformers import pipeline
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score
import wandb

# Initialize W&B
wandb.init(project="codereview-eval")

# Load your model
pipe = pipeline(
    "text-generation",
    model="YOUR_USERNAME/codereview-7b-lora",
    device_map="auto"
)

# Load test set
test_data = load_dataset("YOUR_USERNAME/code-review-processed", split="test")

# Generate predictions
predictions = []
references = []

for example in test_data.select(range(100)):  # Test on 100 examples
    prompt = example["prompt"]
    reference = example["completion"]
    
    output = pipe(prompt, max_length=512, do_sample=True, temperature=0.7)
    prediction = output[0]["generated_text"].replace(prompt, "").strip()
    
    predictions.append(prediction)
    references.append(reference)

# Create evaluation metrics
results = {
    "prompt": [ex["prompt"] for ex in test_data.select(range(100))],
    "prediction": predictions,
    "reference": references,
}

df = pd.DataFrame(results)

# Log to W&B
table = wandb.Table(dataframe=df)
wandb.log({"predictions": table})

# Manual evaluation (rate 50 examples)
print("Rate the following predictions (1-5):")
scores = []

for i in range(50):
    print(f"\n--- Example {i+1} ---")
    print(f"Code:\n{test_data[i]['prompt'][:200]}...")
    print(f"\nPrediction:\n{predictions[i]}")
    print(f"\nReference:\n{references[i]}")
    
    score = int(input("Score (1-5): "))
    scores.append(score)

avg_score = sum(scores) / len(scores)
print(f"\nAverage Score: {avg_score:.2f}/5.0")

wandb.log({"manual_eval_score": avg_score})
wandb.finish()
```

**Metrics to Track:**
- Response quality (1-5 rating)
- Response length appropriateness
- Code understanding accuracy
- Suggestion relevance

**Deliverable Checklist:**
- [ ] 100+ test examples evaluated
- [ ] Metrics logged to W&B
- [ ] Model performance documented
- [ ] Comparison with base model

---

## Phase 2: Deployment (Weeks 5-8)

### 🎯 Goals
- Deploy model to production
- Create user-friendly interface
- Add RAG capabilities
- Integrate with GitHub

### 📊 Deliverables
- [ ] Live demo on HuggingFace Spaces
- [ ] RAG system with code patterns
- [ ] GitHub bot for PR reviews
- [ ] Public portfolio piece

---

### Week 5-6: HuggingFace Spaces Deployment

#### Day 22-28: Gradio Interface (10 hours)

**Create `app.py` in your repo:**

```python
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Load model
model_name = "codellama/CodeLlama-7b-hf"
adapter_name = "YOUR_USERNAME/codereview-7b-lora"

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def review_code(code_snippet, language="python"):
    """Generate code review"""
    prompt = f"""Review the following {language} code:

```{language}
{code_snippet}
```

Provide a constructive code review:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    review = response.replace(prompt, "").strip()
    
    return review

# Example code snippets
examples = [
    ["""def calculate_total(items):
    total = 0
    for item in items:
        total = total + item['price']
    return total""", "python"],
    
    ["""function fetchData() {
    fetch('/api/data')
        .then(response => response.json())
        .then(data => console.log(data))
}""", "javascript"],
]

# Gradio interface
demo = gr.Interface(
    fn=review_code,
    inputs=[
        gr.Textbox(lines=10, label="Code Snippet", placeholder="Paste your code here..."),
        gr.Dropdown(["python", "javascript", "java", "cpp"], label="Language", value="python")
    ],
    outputs=gr.Textbox(lines=10, label="Code Review"),
    title="🤖 CodeReviewAI",
    description="AI-powered code review assistant. Paste your code and get constructive feedback!",
    examples=examples,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
```

**Create `requirements.txt` for Spaces:**

```
torch
transformers
peft
bitsandbytes
accelerate
gradio
```

**Create `README.md` for Spaces:**

```markdown
---
title: CodeReviewAI
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# CodeReviewAI

AI-powered code review assistant built with CodeLlama-7B.

## Features
- Instant code review feedback
- Supports multiple languages
- Constructive suggestions
- Best practices recommendations
```

**Deploy to HuggingFace Spaces:**

1. Go to huggingface.co/spaces
2. Click "Create new Space"
3. Name: codereview-ai
4. SDK: Gradio
5. Connect to GitHub repo
6. Auto-deploys on push!

**Deliverable Checklist:**
- [ ] Gradio app working locally
- [ ] App deployed to HF Spaces
- [ ] Public URL accessible
- [ ] Examples working correctly

---

### Week 7: RAG Enhancement

#### Day 29-35: Add Retrieval System (8 hours)

**Create `src/rag_system.py`:**

```python
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

class CodeRAGSystem:
    def __init__(self):
        # Initialize ChromaDB
        self.client = chromadb.Client()
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create collection
        self.collection = self.client.create_collection(
            name="code_patterns",
            embedding_function=self.embedding_fn
        )
        
        self.load_patterns()
    
    def load_patterns(self):
        """Load common code patterns and best practices"""
        patterns = [
            {
                "id": "pattern_1",
                "code": "list comprehension",
                "description": "Use list comprehensions for simple transformations",
                "example": "[x*2 for x in items]",
                "advice": "More Pythonic and often faster than loops"
            },
            {
                "id": "pattern_2",
                "code": "context manager",
                "description": "Use context managers for resource management",
                "example": "with open('file.txt') as f:",
                "advice": "Ensures proper cleanup of resources"
            },
            # Add more patterns...
        ]
        
        for pattern in patterns:
            self.collection.add(
                documents=[pattern["description"]],
                metadatas=[pattern],
                ids=[pattern["id"]]
            )
    
    def retrieve_relevant(self, code_snippet, n=3):
        """Retrieve relevant patterns for code"""
        results = self.collection.query(
            query_texts=[code_snippet],
            n_results=n
        )
        
        return results["metadatas"][0]
    
    def enhance_review(self, code, base_review):
        """Enhance review with RAG context"""
        patterns = self.retrieve_relevant(code)
        
        context = "\n".join([
            f"- {p['description']}: {p['advice']}"
            for p in patterns
        ])
        
        enhanced = f"""{base_review}

### Additional Best Practices:
{context}
"""
        return enhanced

# Usage in app.py
rag = CodeRAGSystem()

def review_code_with_rag(code_snippet, language="python"):
    # Get base review
    base_review = review_code(code_snippet, language)
    
    # Enhance with RAG
    enhanced_review = rag.enhance_review(code_snippet, base_review)
    
    return enhanced_review
```

**Update `app.py` to use RAG:**

```python
# Add at top
from src.rag_system import CodeRAGSystem

rag = CodeRAGSystem()

# Modify review function
def review_code_enhanced(code_snippet, language="python"):
    base_review = review_code(code_snippet, language)
    return rag.enhance_review(code_snippet, base_review)

# Update interface to use review_code_enhanced
```

**Deliverable Checklist:**
- [ ] RAG system implemented
- [ ] Pattern database populated
- [ ] Reviews enhanced with context
- [ ] Tested on various code examples

---

### Week 8: GitHub Integration

#### Day 36-42: GitHub Bot (8 hours)

**Create `.github/workflows/code_review.yml`:**

```yaml
name: AI Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
        with:
          fetch-depth: 0
      
      - name: Get changed files
        id: changed-files
        uses: tj-actions/changed-files@v40
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install requests
      
      - name: Review code
        env:
          HF_SPACE_URL: ${{ secrets.HF_SPACE_URL }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          python .github/scripts/review_pr.py
```

**Create `.github/scripts/review_pr.py`:**

```python
import os
import requests
import json

def get_pr_diff():
    """Get PR diff from GitHub"""
    github_token = os.getenv('GITHUB_TOKEN')
    repo = os.getenv('GITHUB_REPOSITORY')
    pr_number = os.getenv('GITHUB_EVENT_PATH')
    
    # Read PR event
    with open(pr_number) as f:
        event = json.load(f)
    
    pr_num = event['pull_request']['number']
    
    # Get diff
    headers = {'Authorization': f'token {github_token}'}
    url = f'https://api.github.com/repos/{repo}/pulls/{pr_num}/files'
    
    response = requests.get(url, headers=headers)
    return response.json()

def review_code_via_api(code):
    """Call HuggingFace Space API"""
    hf_space_url = os.getenv('HF_SPACE_URL')
    
    response = requests.post(
        f'{hf_space_url}/api/predict',
        json={'data': [code, 'python']}
    )
    
    return response.json()['data'][0]

def post_review_comment(file_path, review):
    """Post review as PR comment"""
    github_token = os.getenv('GITHUB_TOKEN')
    repo = os.getenv('GITHUB_REPOSITORY')
    
    with open(os.getenv('GITHUB_EVENT_PATH')) as f:
        event = json.load(f)
    
    pr_number = event['pull_request']['number']
    
    comment = f"""## 🤖 AI Code Review

**File:** `{file_path}`

{review}

---
*Powered by CodeReviewAI*
"""
    
    headers = {
        'Authorization': f'token {github_token}',
        'Content-Type': 'application/json'
    }
    
    url = f'https://api.github.com/repos/{repo}/issues/{pr_number}/comments'
    
    requests.post(url, headers=headers, json={'body': comment})

# Main execution
if __name__ == '__main__':
    files = get_pr_diff()
    
    for file in files[:3]:  # Review first 3 files
        if file['filename'].endswith('.py'):
            patch = file.get('patch', '')
            if patch:
                review = review_code_via_api(patch)
                post_review_comment(file['filename'], review)
```

**Deliverable Checklist:**
- [ ] GitHub Action configured
- [ ] Bot posts reviews on PRs
- [ ] Comments formatted nicely
- [ ] Rate limiting handled

---

## Phase 3: Advanced Features (Weeks 9-12)

### 🎯 Goals
- Add monitoring and analytics
- Optimize performance
- Create documentation
- Polish portfolio

### 📊 Deliverables
- [ ] W&B monitoring dashboard
- [ ] Performance optimizations
- [ ] Complete documentation
- [ ] Portfolio-ready project

---

### Week 9-10: Monitoring & Analytics

#### Day 43-56: Comprehensive Monitoring (12 hours)

**Create `src/monitoring.py`:**

```python
import wandb
from datetime import datetime
import json

class ReviewMonitor:
    def __init__(self, project_name="codereview-production"):
        self.run = wandb.init(project=project_name, job_type="inference")
        self.metrics = []
    
    def log_review(self, code, review, language, latency, user_feedback=None):
        """Log each review to W&B"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "language": language,
            "code_length": len(code),
            "review_length": len(review),
            "latency_seconds": latency,
            "user_feedback": user_feedback,
        }
        
        self.metrics.append(log_data)
        wandb.log(log_data)
    
    def log_daily_summary(self):
        """Create daily summary"""
        if not self.metrics:
            return
        
        summary = {
            "total_reviews": len(self.metrics),
            "avg_latency": sum(m["latency_seconds"] for m in self.metrics) / len(self.metrics),
            "languages": dict((m["language"], 0) for m in self.metrics),
        }
        
        wandb.log({"daily_summary": summary})
        
        # Create visualization
        table = wandb.Table(
            columns=["timestamp", "language", "latency", "feedback"],
            data=[[m["timestamp"], m["language"], m["latency_seconds"], m["user_feedback"]] 
                  for m in self.metrics]
        )
        
        wandb.log({"reviews_table": table})

# Usage in app.py
monitor = ReviewMonitor()

def review_code_monitored(code_snippet, language="python"):
    import time
    start = time.time()
    
    review = review_code_with_rag(code_snippet, language)
    
    latency = time.time() - start
    monitor.log_review(code_snippet, review, language, latency)
    
    return review
```

**Create W&B Dashboard:**

1. Go to wandb.ai
2. Create new project: "codereview-production"
3. Add visualizations:
   - Reviews per day (line chart)
   - Average latency (gauge)
   - Language distribution (pie chart)
   - User feedback trends (bar chart)

**Deliverable Checklist:**
- [ ] All reviews logged to W&B
- [ ] Dashboard created
- [ ] Metrics tracked continuously
- [ ] Alerts configured

---

### Week 11: Performance Optimization

#### Day 57-63: Speed & Cost Optimization (8 hours)

**Create `src/optimization.py`:**

```python
import torch
from functools import lru_cache
import hashlib

class OptimizedInference:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = {}
    
    @lru_cache(maxsize=100)
    def cached_review(self, code_hash):
        """Cache reviews for identical code"""
        if code_hash in self.cache:
            return self.cache[code_hash]
        return None
    
    def quantize_model(self):
        """Apply dynamic quantization"""
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
