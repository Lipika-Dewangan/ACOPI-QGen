# ACOPI-QGen: A Generative Framework for Implicit Sentiment Quintuple Extraction

ACOPI-QGen is a novel end-to-end generative model designed to jointly extract all five sentiment elements—**Aspect Term, Aspect Category, Opinion Term, Sentiment Polarity, and Implicitness**—from user-generated content, especially in the presence of figurative or implicit expressions.

---

## 🚀 Overview

Traditional ABSA methods often focus on extracting pairs, triplets, or quadruples and typically struggle with **implicit sentiment reasoning**. ACOPI-QGen reformulates this task as a **structured quintuple extraction problem**, introducing a **multi-class implicitness classification scheme** using the following four categories:

- `EAEO`: Explicit Aspect & Explicit Opinion  
- `EAIO`: Explicit Aspect & Implicit Opinion  
- `IAEO`: Implicit Aspect & Explicit Opinion  
- `IAIO`: Implicit Aspect & Implicit Opinion

---

## 🧠 Model Architecture

ACOPI-QGen consists of the following key components:

- **BERT Encoder**: Encodes contextual token representations.
- **Supervised Contrastive Learning (SCL)**: Four projection heads for aspect, opinion, sentiment, and implicitness roles to enhance semantic separation.
- **Relational Graph Attention Network (RGAT)**: Injects syntactic relational bias using a Surrogate Aspect-Opinion Dependency Tree (SAODT).
- **Non-Autoregressive Decoder**: Decodes complete sentiment quintuples in parallel, avoiding the limitations of sequential decoding.

---

## 📦 Dataset: ACOPI

**ACOPI dataset** across three domains
- Restaurant-ACOPI  
- Laptop-ACOPI 
- Shoes-ACOPI 

Synthetic implicit samples were added using Large Language Models (LLMs) like **GPT-4o** and **LLaMA-3-70B** to improve class balance and diversity.
 📌 Note: The full annotated datasets will be made publicly available upon acceptance of the paper.
---

## 📊 Performance Summary

ACOPI-QGen achieves significant improvements in quintuple extraction across domains:

| Dataset         | F1 Score (%) |
|------------------|--------------|
| Restaurant-ACOPI | 55.68        |
| Laptop-ACOPI     | 48.43        |
| Shoes-ACOPI      | 60.49        |

Compared to SOTA LLMs and supervised baselines, ACOPI-QGen demonstrates superior performance on both explicit and implicit sentiment components.
acopi-qgen/
│
├── data/               # Contains ACOPI datasets (Restaurant, Laptop, Shoes)
├── models/             # Model architecture components
├── utils/              # Utilities for preprocessing, metrics, etc.
├── train.py            # Training pipeline
├── evaluate.py         # Evaluation pipeline
├── README.md           # This file
└── requirements.txt    # Python dependencies


