This repository provides the implementation of ACOPI-QGen, an end-to-end generative framework designed for fine-grained implicit Aspect-Based Sentiment Analysis (ABSA). The model jointly extracts all five sentiment components—Aspect Term, Aspect Category, Opinion Term, Sentiment Polarity, and Implicitness—as structured quintuples from input text. Implicitness is modeled as a multi-class classification task with four explicit-implicit combinations: EAEO, EAIO, IAEO, IAIO, advancing beyond the binary framing used in prior work.

The architecture integrates:

BERT-based contextual encodings for semantic understanding,

Fine-grained Supervised Contrastive Learning (SCL) with role-specific projection heads for enhanced alignment,

Relational Graph Attention Network (RGAT) guided by a Surrogate Aspect-Opinion Dependency Tree (SAODT) for syntactic bias modeling,

and a non-autoregressive decoder for parallel generation of sentiment quintuples, improving inference efficiency and minimizing exposure bias.
