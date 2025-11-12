# vLLM: Efficient Memory Management for LLM Serving with PagedAttention  
**Paper**: https://arxiv.org/pdf/2309.06180  
**Authors**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica  
**Presenter**: Yan Zhang  

---

# 1. Overview

Large Language Models (LLMs) generate tokens sequentially, and for each token, they must repeatedly read a large **Key–Value (KV) cache** stored in GPU memory.  
This KV cache quickly becomes the **dominant memory bottleneck**, limiting batch size and throughput.

The paper introduces **PagedAttention**, a memory-efficient attention mechanism inspired by **operating-system paging**.  
It splits KV cache into fixed-size **blocks**, stored non-contiguously, and managed via a **block table**.

This design allows:
- **2×–4× higher throughput**  
- **Up to 3× less memory waste**  
- **Near-zero fragmentation**  

---

# 2. Problem: Why LLM Serving Is Inefficient

## 2.1 KV Cache Dominates Memory

During autoregressive generation:
- Each token attends to all previous tokens  
- Model stores all past tokens’ K/V vectors in GPU memory  
- For OPT-13B, ~800 KB per token  
- A 2048-token sequence → **up to 1.6 GB** KV cache  

**Only a few requests can fit on one GPU.**

> **Paper source: Figure 1.**

*(Insert Figure 1: GPU memory breakdown)*  
`![Figure 1: GPU Memory Breakdown](figures/figure1.png)`

---

## 2.2 Why Existing Systems Waste Memory

Most systems allocate **one large contiguous KV array per request**  
(size = max sequence length).

This causes three forms of waste:

### ✔ Reservation Waste  
Large chunks reserved for future tokens remain empty for most of inference.

### ✔ Internal Fragmentation  
A request finishes early but its large block stays mostly unused.

### ✔ External Fragmentation  
Mixed-size requests leave “holes” too small to reuse.

> **Paper result:** Only **20.4%–38.2%** of KV cache stores valid token data.  
> **(Figure 2 in the paper.)**

*(Insert Figure 2: KV cache waste breakdown)*  
`![Figure 2: KV Cache Fragmentation](figures/figure2.png)`

---

# 3. Key Idea: PagedAttention

PagedAttention borrows the logic of **virtual memory / paging**.

## 3.1 KV Cache → Fixed-Size Blocks

Instead of a giant contiguous buffer:

- KV cache is split into **blocks** (e.g., 16 tokens/block)  
- Blocks can be placed **anywhere** in GPU memory  
- A per-request **block table** maps logical order → physical blocks  

### Conceptual structure:

```
Logical KV Blocks:   [B0][B1][B2][B3] ...
Physical Storage:    [B1]   [B3][B0]   [B2] ...
```

> **Paper source: Figure 5.**

*(Insert Figure 5: logical blocks → physical blocks)*  
`![Figure 5: Logical to Physical Block Mapping](figures/figure5.png)`

---

## 3.2 Benefits

| Problem in Existing Systems | How PagedAttention Fixes It |
|-----------------------------|------------------------------|
| Huge contiguous buffers     | Allocate blocks **on-demand** |
| Internal fragmentation      | Only last block partially wasted |
| External fragmentation      | Uniform block size avoids holes |
| Prefix duplication          | Blocks can be **shared** |

---

# 🎤 Audience Question 1  
(Required by rubric)

<details>  
<summary><strong>Why does allocating one large contiguous buffer cause so much memory waste?</strong></summary>

Because the system must pre-reserve memory for the **maximum sequence length**, and these large chunks cannot be compacted or partially reused.  
As soon as sequences finish early or vary in length, they leave large unused regions (internal fragmentation) and holes (external fragmentation).  

PagedAttention avoids this by allocating **small uniform blocks**, not giant arrays.  
</details>

---
