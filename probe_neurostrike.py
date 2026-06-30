import sys, platform
print("Python version:", sys.version)
print("Platform:", platform.platform())

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from scipy.stats import zscore
import os
import json
from pathlib import Path

# to use gpu if available
device = 'auto'

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

# get the model
model_name = "meta-llama/Llama-3.2-1B-Instruct"
attn_implementation = 'eager'
dataset = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
split = "train"
print(f"=====Tested Model: {model_name}=====")

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map=device, attn_implementation=attn_implementation, trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
#tokenizer.padding_side  = 'left'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id
device = model.device
#count mlp layers
mlp_count = 0
for name, module in model.named_modules():
  if any(keyword in name.lower() for keyword in ['gate', 'up']):
    # print(name)
    mlp_count += 1
    # These two model has gate and up fused into one single layer
    if model_name.lower().startswith("phi-4") or model_name.lower().startswith("dna"):
       mlp_count = mlp_count
    else:
        mlp_count= int(mlp_count/2)

print(mlp_count)

# Config for safety neuron extraction
num_responses = 1
num_repeat_training = 1
safe_neuron_threshold = 3

# Max new tokens for the inference. Set it to 256 to speed up the inference
max_new_tokens = 128

ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")
data = ds["train"]
all_prompts = data["prompt"]
all_labels = data["prompt_label"]
categories = data["violated_categories"]

print(f"Number of all prompts: {len(all_prompts)}")
expanded_labels = []
expanded_prompts = []
expanded_categories = []
for prompt, label, category in zip(all_prompts, all_labels, categories):
    expanded_prompts.extend([prompt]*num_responses)
    expanded_labels.extend([label]*num_responses)
    expanded_categories.extend([category]*num_responses)

print(f'Number of expanded prompts: {len(expanded_prompts)}')

def parse_categories(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]
    s = str(x).strip()
    if s == "" or s.lower() == "null" or s.lower() == "none":
        return []
    for sep in ["|", ";", ","]:
        if sep in s: 
            parts = [p.strip() for p in s.split(sep)]
            return [p for p in parts if p]
    return [s]

violated_lists = []
for label, cat in zip(expanded_labels, expanded_categories):
    if str(label).lower() == "unsafe":
        violated_lists.append(parse_categories(cat))
    else: 
        violated_lists.append([])
all_cats = sorted({c for cats in violated_lists for c in cats})
cat_to_idx = {c: i for i, c in enumerate(all_cats)}

N = len(expanded_prompts)
C = len(all_cats)
Y = np.zeros((N,C), dtype=np.float32)
for i, cats in enumerate(violated_lists):
    for c in cats:
        Y[i, cat_to_idx[c]] = 1.0

# make prompts from the expanded prompts
def construct_prompt(tokenizer, expanded_prompts, system_prompt=None):
    prompts = []

    # Default prompt if its empty
    if system_prompt is None:
        system_prompt = ("You are an AI that provides direct and precise answers." "Respond only to the question without additional details or explanations.")

    for question in expanded_prompts:
        chat = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question},]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    return prompts

def batchify(n, batch_size):
    """Yield successive batches from list."""
    for start in range(0, n, batch_size):
        yield list(range(start, min(n, start + batch_size)))

def collect_pooled_activations(model, tokenizer, prompts, subset, pool="max", batch_size=4,):
    """
    Returns:acts_np: dict[layer_name] -> np.ndarray [N, H] where H is the FFN (post-gating) neuron dimension
    """
    activations = {layer: [] for layer in subset.keys()}

    def make_hook(layer_name):
        keep_idx = subset[layer_name]
        def hook(module, input, output):
            # inputs[0] is post-gating FFN activation: [B, T, H]
            if pool == "mean":
                pooled = output.mean(dim=1)
            elif pool == "max":
                pooled = output.max(dim=1)[0]
            else:
                raise ValueError("pool must be 'mean' or 'max'")
            pooled_subset = pooled[:, keep_idx]
            activations[layer_name].append(pooled_subset.detach().cpu().to(torch.float16).numpy().astype(np.float16, copy=False))
            return None
        return hook

    handles = []
    for name, module in model.named_modules():
        if name in subset:
            handles.append(module.register_forward_hook(make_hook(name)))

    for idx_batch in tqdm(list(batchify(len(prompts), batch_size))):
        batch_prompts = [prompts[i] for i in idx_batch]
        input_tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=128,).to(model.device)

        with torch.no_grad():
            _ = model(**input_tokens)

    for h in handles:
        h.remove()
    print("DEBUG layer names")
    for k in activations.keys():
        print(k)
    return {layer: np.concatenate(chunks, axis=0) for layer, chunks in activations.items()}

# for probing
def filter_activations_by_subset(activations, subset):
    filtered={}
    for layer, acts in activations.items():
        if layer not in subset:
            continue
        idx = subset[layer]
        filtered[layer] = acts[:, idx]
    return filtered

# train head on pooled activations, freeze it for attribution
class SafetyHead(nn.Module):
  def __init__(self, hidden_dim, num_classes):
    super().__init__()
    self.linear = nn.Linear(hidden_dim, num_classes)

  def forward(self, x):
    return self.linear(x)

def compute_pos_weight(y):
    pos = y.sum(dim=0)
    pos = torch.clamp(pos, min=1.0)
    neg = y.shape[0] - pos
    return neg / pos

# train the head, using pooled acitvations as features
def train_head(activations_by_layer, labels, device, lr=1e-3, weight_decay=1e-3, epochs=200):
    y = torch.tensor(labels, dtype=torch.float32,device=device)
    C = Y.shape[1]
    pos_weight = compute_pos_weight(y).to(device)  
    bce = nn.BCEWithLogitsLoss(pos_weight = pos_weight)

    heads = {}
    N = labels.shape[0]
    idx_all = np.arange(N)
    for layer_name, x_np in activations_by_layer.items():
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        h = x.shape[1]
        head = SafetyHead(h, C).to(device)
        opt = optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(epochs):
            head.train()
            opt.zero_grad()
            logits = head(x)
            loss = bce(logits, y)
            loss.backward()
            opt.step()

        head.eval()
        for p in head.parameters():
            p.requires_grad_(False) #freeze head

        heads[layer_name] = head.eval()

    return heads

prompts = construct_prompt(tokenizer, expanded_prompts)

subset_path = " " # Fill in subset path
with open (subset_path, "r") as f:
    subset_payload = json.load(f)

if "safety_neurons" in subset_payload:
    neurostrike_neurons = subset_payload["safety_neurons"]
else:
    neurostrike_neurons = subset_payload

activations = collect_pooled_activations(model=model, tokenizer=tokenizer, prompts=prompts, subset=neurostrike_neurons, pool="max")
#activations = filter_activations_by_subset(activations, neurostrike_neurons)
subset_mapping = {}
for layer, idxs in neurostrike_neurons.items():
    subset_mapping[layer] = {i: int(j) for i, j in enumerate(idxs)}

labels = Y
print("prompts:", len(prompts))
print("labels:", len(labels))
print("activations:", len(activations))

# train neurostrike head (instead of the neurostrike probe)
heads = train_head(activations_by_layer=activations, labels=labels,
                   device=model.device, lr=1e-3, weight_decay=1e-3,
                   epochs=200)

for layer, head in heads.items():
    W = head.linear.weight.detach().cpu().numpy()
    mapping = subset_mapping[layer]
    print(f"\nLayer: {layer}")
    for i_local in range(W.shape[1]):
        i_global = mapping[i_local]
        weights = W[:, i_local]
        rounded_weights = np.round(weights, 4)
        print(f"Neuron {i_global}: weight={rounded_weights}")

#combine neurons with their category weights
neuron_category_mapping = {}
for layer, head in heads.items():
    W = head.linear.weight.detach().cpu().numpy()
    subset_indices = neurostrike_neurons[layer]
    neuron_category_mapping[layer] = {}
    for j_local, j_global in enumerate(subset_indices):
        weights = W[:, j_local]
        pairs = [(int(subset_indices[j_local]), float(weights[c_idx])) for c_idx in range(len(all_cats))]
        pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
        neuron_category_mapping[layer][int(j_global)] = pairs


#pick the top 3 category weights per neuron
top_k = 5
neuron_topk_categories = {}
for layer, head in heads.items():
    W = head.linear.weight.detach().cpu().numpy()
    subset_indices = neurostrike_neurons[layer]
    neuron_topk_categories[layer] = {}
    for j_local, j_global in enumerate(subset_indices):
        weights = W[:, j_local]
        topk_idx = np.argsort(np.abs(weights))[-top_k:][::-1]
        topk = [{"category": all_cats[c_idx], "weight": float(weights[c_idx])} for c_idx in topk_idx]
        neuron_topk_categories[layer][int(j_global)] = topk

# get all scores 
neuron_all_categories = {}
for layer, head in heads.items():
    W = head.linear.weight.detach().cpu().numpy()
    subset_indices = neurostrike_neurons[layer]
    neuron_all_categories[layer] = {}
    for j_local, j_global in enumerate(subset_indices):
        weights = W[:, j_local]
        neuron_all_categories[layer][int(j_global)] = {all_cats[c_idx]: float(weights[c_idx]) for c_idx in range(len(all_cats))}

# Convert numpy arrays to lists for JSON serialization
json_compatible_neurons = {}
for layer, indices in neurostrike_neurons.items():
    json_compatible_neurons[layer] = list(indices)

# Save the results to a JSON file
model_id = 0
out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)
job_id = os.environ.get("SLURM_JOB_ID", "local")
out_path = out_dir / f"probed_safety_neurons_model{model_id}{job_id}.json"

payload = {
    "method": "activation",
    "model_id": model_id,
    "model_name": model_name,
    "neuron_all_categories": neuron_all_categories}

with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)

print(f"[OK] Saved safety neurons to {out_path}")
