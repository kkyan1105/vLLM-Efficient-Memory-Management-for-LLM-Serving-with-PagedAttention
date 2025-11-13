# vLLM: Efficient Memory Management for LLM Serving with PagedAttention  
**Paper**: https://arxiv.org/pdf/2309.06180  
**Authors**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica  
**Presenter**: Yan Zhang  

---

## 1. Overview

Large language model inference is dominated not by compute, but by GPU memory usage — especially the memory consumed by the **Key–Value (KV) cache** during autoregressive decoding. For each generated token, the model must use all previously computed keys and values. This causes the KV cache to grow linearly with sequence length, and for models such as OPT-13B and LLaMA-13B, a single request can consume over **1 GB** of KV-cache memory.

To illustrate the magnitude of KV-cache memory, the table below shows the memory breakdown for 13B/66B/175B OPT models. Even a 13B model requires **12 GB** of KV-cache memory at maximum capacity.

<p align="center"> <img src="figs/table1.png" width="65%"> </p>

Traditional LLM serving systems allocate **one large contiguous KV buffer** per request. This design is simple but extremely inefficient. As requests with different lengths start and finish, GPU memory forms small free gaps that cannot hold another contiguous KV buffer. Even with plenty of total remaining memory, the system rejects new requests due to external **fragmentation**. Empirical results show that only **20–38%** of allocated KV memory contains real token data.

<p align="center"> <img src="figs/figure1.png" width="65%"> </p>

To address this mismatch, vLLM introduces **PagedAttention**, inspired by OS virtual memory. Instead of requiring contiguous memory, vLLM splits KV-cache into fixed-size blocks. A per-request **block table** maps logical positions to physical locations. Blocks can be placed anywhere in GPU memory, eliminating fragmentation, improving block reuse, enabling continuous batching, and significantly increasing throughput.

