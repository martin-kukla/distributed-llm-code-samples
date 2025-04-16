# Distributed LLM training: code samples

The code samples on how to distribute the LLM training between GPUs/nodes. The code samples are written from the first principle.

# Files
- **distributed_ffns.py**: shows how to distribute the computation of Transformer's FFN sublocks among GPUs/nodes (currently implemented: DDP and FSDP).
