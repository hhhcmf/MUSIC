# MUSIC: Multi-hop Reasoning via Modality-specific Inference Collaboration for Multi-modal QA

This is the official implementation of our paper, **"MUSIC: Multi-hop Reasoning via Modality-specific Inference Collaboration for Multi-modal QA"**.

This framework is designed to tackle complex Multi-modal Question Answering (MMQA) problems. It achieves this by constructing a unified knowledge graph and designing a multi-agent collaborative system that supports multi-hop reasoning over heterogeneous data sources, including tables, text, and images.

## System Architecture

The core pipeline of MUSIC is divided into two phases: **Offline Construction** and **Online Inference**.

## Getting Started

### 1. Prerequisites

#### 1.1 Environment Setup

We recommend using Conda to manage the environment for consistency.

```bash
# Create and activate the Conda environment
conda create -n music python=3.11 -y
conda activate music

# Install all dependencies
pip install -r requirements.txt
```

#### 1.2 Configuration
Before running any scripts, please configure your local paths.
Config file path: src/config/config.yaml
### 2. Offline Processing
The goal of the offline phase is to construct and pre-process the global knowledge graph required for online inference.

#### 2.1 Build the Global Knowledge Graph
This step iterates through all source data to build a unified multi-modal knowledge graph.
```bash
python -m src.offline_process.kg_construct
```
#### 2.2 Train Global KG Embeddings
This step uses the constructed graph to train a Knowledge Graph Embedding model.
```bash
python -m src.KGE.train
```
### 3. Online Inference
After completing all offline steps, you can run the online inference system to answer complex questions.

#### 3.1 Run the Inference Service
```bash
python -m src.online_query.main
```