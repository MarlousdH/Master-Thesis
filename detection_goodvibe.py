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

# to use gpu if available
device = 'auto'

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

# get the model
model_name = "meta-llama/Llama-3.2-1B-Instruct"
attn_implementation = 'eager'
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

# get the datasets, label 0=benign, label 1=malicious
prompts = []
labels = []
ds = load_dataset("walledai/CatHarmfulQA")
prompts += ds['en']['prompt'] #en/ch/ve
labels += [1] * len(ds['en']['prompt'])
ds = load_dataset("declare-lab/HarmfulQA")
prompts += ds['train']['question']
labels += [1] * len(ds['train']['question'])
ds = load_dataset("LLM-LAT/harmful-dataset")
prompts += ds['train']['prompt']
labels += [1] * len(ds['train']['prompt'])
print(f'Number of malicious prompts: {len(prompts)}')
all_prompts = prompts
all_labels = labels
# Benign dataset
ds = load_dataset("facebook/natural_reasoning")
all_prompts += ds['train']['question'][:len(all_labels)]
all_labels += [0] * len(all_labels)

print(f'Number of all prompts: {len(all_prompts)}')
expanded_labels = []
expanded_prompts = []
for prompt, label in zip(all_prompts, all_labels):
  expanded_prompts.extend([prompt] * num_responses)
  expanded_labels.extend([label] * num_responses)

print(f'Number of expanded prompts: {len(expanded_prompts)}')

# make prompts from the expanded prompts
def construct_prompt(tokenizer, expanded_prompts, system_prompt=None):
    prompts = []

    # Default prompt if its empty
    if system_prompt is None:
        system_prompt = ("You are an AI that provides direct and precise answers. " "Respond only to the question without additional details or explanations.")

    for question in expanded_prompts:
        chat = [{"role": "system", "content": system_prompt},{"role": "user", "content": question},]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    return prompts

def batchify(n, batch_size):
    """Yield successive batches from list."""
    for start in range(0, n, batch_size):
        yield list(range(start, min(n, start + batch_size)))

def collect_pooled_activations(model, tokenizer, prompts, pool="mean", batch_size=4,):
    """
    Returns:acts_np: dict[layer_name] -> np.ndarray [N, H] where H is the FFN (post-gating) neuron dimension
    """
    activations = {}

    def make_prehook(layer_idx):
        def prehook(module, inputs):
            # inputs[0] is post-gating FFN activation: [B, T, H]
            x = inputs[0]
            if pool == "mean":
                pooled = x.mean(dim=1)
            elif pool == "max":
                pooled = x.max(dim=1)[0]
            else:
                raise ValueError("pool must be 'mean' or 'max'")
            activations.setdefault(layer_idx, []).append(pooled.detach().cpu().float().numpy())
            return None
        return prehook

    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_prehook(f"layer{layer_idx}.ffn")))

    for idx_batch in tqdm(list(batchify(len(prompts), batch_size))):
        batch_prompts = [prompts[i] for i in idx_batch]
        input_tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True,max_length=128,).to(model.device)
        with torch.no_grad():
            _ = model(**input_tokens)

    for h in handles:
        h.remove()
    print("DEBUG layer names")
    for k in activations.keys():
        print(k)
    return {layer: np.concatenate(chunks, axis=0) for layer, chunks in activations.items()}

# train head on pooled activations, freeze it for attribution
class SafetyHead(nn.Module):
  def __init__(self, hidden_dim):
    super().__init__()
    self.linear = nn.Linear(hidden_dim, 1)

  def forward(self, x):
    return self.linear(x)

# train the head, using pooled acitvations as features
def train_head(activations_by_layer, labels, device, lr=1e-3, weight_decay=1e-3, epochs=200):
  y = torch.tensor(labels, dtype=torch.float32,device=device).view(-1,1)
  bce = nn.BCEWithLogitsLoss()

  heads = {}
  for layer_name, x_np in activations_by_layer.items():
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    h = x.shape[1]
    head = SafetyHead(h).to(device)
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

def goodvibe_grad_importance(model, tokenizer, prompts, labels, heads_by_layer, pool="mean", batch_size=4):
    """
    Returns:
      importance_by_layer: dict[layer] -> np.ndarray [H]
                           mean over dataset of |dL/d(pooled_activation)|
    """
    bce = nn.BCEWithLogitsLoss()
    activations_pool = {}  # layer_name -> pooled tensor [B,H] (current batch)

    def make_grad_hook(layer_idx):
        def hook(module, inputs):
            x = inputs[0]
            if pool == "max":
                pooled = x.max(dim=1)[0]   # [B,H]
            elif pool == "mean":
                pooled = x.mean(dim=1)     # [B,H]
            else:
                raise ValueError("pool must be 'max' or 'mean'")

            pooled.retain_grad()
            activations_pool[layer_idx] = pooled
            return None
        return hook

    # Register hooks
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_grad_hook(f"layer{layer_idx}.ffn")))

    imp_sum = {}   # layer -> torch[H]
    n_seen = 0

    N = len(prompts)
    for idx_batch in tqdm(list(batchify(N, batch_size))):
        batch_prompts = [prompts[i] for i in idx_batch]
        y = torch.tensor([labels[i] for i in idx_batch],
                         dtype=torch.float32, device=model.device).view(-1, 1)

        input_tokens = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(model.device)

        model.zero_grad(set_to_none=True)
        activations_pool.clear()

        _ = model(**input_tokens)  # fills activations_pool

        # Compute supervised loss across layers with trained heads
        loss_terms = []
        for layer_idx, pooled in activations_pool.items():
            head = heads_by_layer.get(layer_idx, None)
            if head is None:
                continue
            # Cast pooled activations to float32 to match head's weights
            logits = head(pooled.to(torch.float32))         # [B,1]
            loss_terms.append(bce(logits, y))

        if not loss_terms:
            raise RuntimeError("No matching heads found for hooked layers.", f"hooked layers:{list(activations_pool.keys())}", f"head layers: {list(heads_by_layer.keys())}")

        loss = torch.stack(loss_terms).mean()
        loss.backward()

        B = len(idx_batch)
        for layer_idx, pooled in activations_pool.items():
            if layer_idx not in heads_by_layer:
                continue
            if pooled.grad is None:
                continue
            per_neuron = pooled.grad.abs().mean(dim=0)  # [H]
            imp_sum[layer_idx] = imp_sum.get(layer_idx, 0) + per_neuron.detach() * B

        n_seen += B

    for h in handles:
        h.remove()

    importance_by_layer = {layer: (imp / n_seen).float().cpu().numpy() for layer, imp in imp_sum.items()}
    return importance_by_layer

def select_neurons_from_importance(importance_by_layer, k):
    selected = {}
    select_importance = {}
#    print("layers gone through: ", importance_by_layer)
    for layer, imp in importance_by_layer.items():
        if len(imp) <= k:
            topk = np.arange(len(imp))
        else:
            topk = np.argsort(imp)[-k:]
            #selected[layer] = np.sort(topk)
            select_importance[layer] = [(i, imp[i]) for i in topk]
#            print("selected neurons and their importance value: ", select_importance)
            print("average importance per layer: ", np.mean(imp[topk]))
        selected[layer] = np.sort(topk)
    return selected, select_importance

prompts = construct_prompt(tokenizer, expanded_prompts)

# collect pooled activations to train the head
activations = collect_pooled_activations(model=model, tokenizer=tokenizer,
                                         prompts=prompts, pool="mean")
labels = expanded_labels
print("prompts:", len(prompts))
print("labels:", len(labels))
print("activations:", len(activations))

# train goodvibe head (instead of the neurostrike probe)
heads = train_head(activations_by_layer=activations, labels=labels,
                   device=model.device, lr=1e-3, weight_decay=1e-3,
                   epochs=200)

# gradient attribution
importance = goodvibe_grad_importance(model=model, tokenizer=tokenizer,
                                      prompts=prompts, labels=labels, heads_by_layer=heads,
                                      pool="mean",
                                      batch_size=4,)

# select neurons
goodvibe_neurons, neuron_importance = select_neurons_from_importance(importance,k=120)
import json

#for pca
selected_neurons = []
for layer_name, idxs in goodvibe_neurons.items():
    for n in idxs:
        selected_neurons.append((layer_name, int(n)))
selected_neurons = sorted(selected_neurons, key=lambda x: (str(x[0]), x[1]))

N = len(labels)
X = []
y = []
for i in range(N):
    feat = []
    for(layer_name, n) in selected_neurons:
        feat.append(float(activations[layer_name][i,n]))
    X.append(feat)
    y.append(int(labels[i]))
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
results = [{"label":int(y[i]), "features": X[i].tolist()} for i in range(N)]

# Convert numpy arrays to lists for JSON serialization
json_compatible_neurons = {}
for layer, indices in goodvibe_neurons.items():
    json_compatible_neurons[layer] = indices.tolist()

# Save the results to a JSON file
from pathlib import Path
import os

model_id = 0
out_dir = Path("outputs/gradient-based")
out_dir.mkdir(parents=True, exist_ok=True)
job_id = os.environ.get("SLURM_JOB_ID", "local")
out_path = out_dir / f"safety_neurons_model{model_id}{job_id}.json"

out_imp = out_dir / f"importance_neurons_model{model_id}{job_id}.json"

payload = {"method": "gradient", "model_id": model_id, "model_name": model_name, "safety_neurons": json_compatible_neurons,  # layer -> list[int]
           }

with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)

imp_json = {}
for layer, neurons in neuron_importance.items():
    imp_json[layer] = [{"neuron": int(i), "score": float(score)} for i, score in neurons]

with open(out_imp, "w") as f:
    json.dump(imp_json, f, indent=2)

with open(f"gradient_pca_{job_id}.json", "w") as f:
    json.dump(results, f)

print(f"[OK] Saved safety neurons to {out_path}")
