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

To address this mismatch, vLLM introduces **PagedAttention**, inspired by OS virtual memory. Instead of requiring contiguous memory, vLLM splits KV-cache into **fixed-size blocks**. A per-request **block table** maps logical positions to physical locations. Blocks can be placed anywhere in GPU memory, eliminating fragmentation, improving block reuse, enabling continuous batching, and significantly increasing throughput.

---

### Question 1
Why does allocating one large contiguous KV cache per request inevitably cause memory fragmentation as batch composition changes over time?
<details>
  <summary><strong>Answer</strong><br>
  </summary>

Because freed memory returns as many small scattered gaps, but each request needs one large continuous KV buffer. New requests cannot fit into these small fragments, so memory becomes unusable even when total free memory is still large.

</details>

## 2. Problem: Why Existing KV Allocation Fails

Traditional LLM serving frameworks such as FasterTransformer or HuggingFace Transformers use a **contiguous KV cache per request**. Although easy to implement, this causes three fundamental types of memory waste.

### 2.1 Reservation Waste

Because the size of the KV cache must support the maximum possible sequence length, the system reserves far more memory than most real requests need. A user input of 200 tokens still reserves space for the maximum length (e.g., 2048 or 4096), leaving a large unused portion.

### 2.2 Internal Fragmentation

If a request finishes earlier than expected, the unused portion of its preallocated KV buffer remains unusable by other requests. Even if 70% of the buffer is empty, no other request can use that space, because buffers must stay contiguous.

### 2.3 External Fragmentation

This is the most damaging form. As requests with different lengths start and finish over time, the GPU memory becomes filled with small free gaps. These gaps cumulatively hold a large amount of free memory, but because each gap is too small to fit a full contiguous KV buffer, they cannot be used.

<p align="center"> <img src="figs/figure3.png" width="65%"> </p>
