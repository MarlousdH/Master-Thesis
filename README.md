# Causal and Representational Analysis of Safety Neuron Identification in Large Language Models
This repository contains the custom code used in my master's thesis on safety neuron identification in large language models.

The repository includes code for:

- Safety neuron detection using NeuroStrike- and GoodVibe-based approaches
- Neuron pruning, Attack Success Rate (ASR) and utility evaluation
- Probing-based analysis of identified safety neurons

## Important

This repository does *not* contain the complete experimental codebase.

The code provided here consists only of the files that were added or modified for the thesis. It should be used together with the original NeuroStrike implementation rather than as a standalone project.

The original NeuroStrike repository can be found here:

- [NeuroStrike](https://github.com/wu-lichao/NeuroStrike-Neuron-Level-Attacks-on-Aligned-LLMs)

Users should first obtain the NeuroStrike codebase and then integrate the files from this repository.

## Repository Contents

````text
detection_goodvibe.py      # GoodVibe-based safety neuron detection
detection_neurostrike.py   # NeuroStrike safety neuron detection
probe_goodvibe.py          # Probing analysis for GoodVibe neurons
probe_neurostrike.py       # Probing analysis for NeuroStrike neurons
prune_and_get_asr.py       # Neuron pruning and ASR evaluation
requirements.txt           # Python dependencies
````

## Installation

Install the required dependencies using:

bash
pip install -r requirements.txt


Additional setup may be required depending on the NeuroStrike version being used.

## Reproducibility

The experiments reported in the thesis were executed on a GPU compute cluster using bash scripts for job scheduling and experiment execution. These bash scripts are not included in this repository.
