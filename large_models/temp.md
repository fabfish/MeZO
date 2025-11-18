Dear Reviewer 4cLS,

Thank you for your continued engagement and valuable feedback. We appreciate the opportunity to address your remaining uncertainty.

#### 1. On Performance Generalization 

We run more extensive experiments to verify that the performance gains of FOCUS generalize with deeper training and across more tasks. Here are the results:

**LLaMA-2-7B Fine-tuning**

We conducted longer experiments on LLaMA-2-7B with a smaller batch size (bs=1 for 1000 steps, which is more common for low-end devices) and a wider hyperparameter search. As shown below, with more training steps, the benefit of the second-order information in FOCUS becomes more pronounced, leading to improved scores over MeZO on tasks like BoolQ and WSC where they were previously tied.

| Model | Batch Size | Steps | Method | RTE | WIC | BoolQ | WSC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LLaMA-2-7B | 16 | 100 | MeZO | 60.6 | 50.2 | 66.0 | 63.5 |
| | | | FOCUS | 61.7 | 50.2 | 66.0 | 63.5 |
| LLaMA-2-7B | 1 | 1000 | MeZO | 60.6 | 50.1 | 66.0 | 59.6 |
| | | | **FOCUS** | **61.7** | **50.3** | **67.5** | **62.5** |

**RoBERTa-350M Fine-tuning**

To further validate that the benefits are not task-specific, we ran more experiments on RoBERTa-350M over 5,000 steps on four different tasks. The results demonstrate a consistent and clear advantage for FOCUS across the board, confirming that its convergence benefits do generalize.

| Model | Steps | Method | MNLI | RTE | SST2 | SNLI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| RoBERTa-350M | 5k | MeZO | 61.7 | 67.5 | 91.6 | 69.2 |
| | | FOCUS | **64.8** | **69.3** | **92.5** | **72.5** |

We acknowledge that ZO methods inherently have high variance, and our hyperparameter search remains finite. However, these new results suggest that under appropriate settings, the second-order information in FOCUS provides a reliable performance improvement over MeZO.

#### 2. Unique Advantages of FOCUS at Scale

Beyond accuracy, we highlight two unique advantages of our BCD-based approach that become critical at scale.

**(1) Significant Speedup in Training Time**

The architectural design of FOCUS leads to stable speed improvements as models and batch sizes grow. On OPT-30B, FOCUS is nearly 2x faster than MeZO and almost 3x faster than HiZOO.

| Model | Method | Accuracy | Time | Notes |
| :--- | :--- | :--- | :--- | :--- |
| OPT-30B | MeZO | 90.6% | 13.7h | 2x A100, bs=32 |
| | HiZOO | 90.3% | 20.8h | 2x A100, bs=32 |
| | FOCUS | **92.9%** | **7.5h** | **1.83x Faster** vs MeZO |
| | FOCUS | **93.6%** | 9.9h | 8x A100, bs=128 |

**(2) Superior Memory Scaling with Larger Batches**

As the batch size increases, FOCUS can become more memory-efficient than MeZO. 
This is because our BCD approach only holds information for the active block, whereas MeZO's memory for optimizer states scales with the full parameter set. 
At a small batch size, FOCUS's memory usage is slightly higher due to its fixed overhead for storing diagonal Hessian information for its active parameter block.
As the batch size increases, FOCUS becomes more memory-efficient because its gradient estimation cost scales only with that small block, whereas MeZO's estimation cost scales with the full model's parameters and thus grows much more rapidly.
This demonstrates a trade-off between fixed overhead and scaling costs.

| Batch Size | Method | Memory |
| :--- | :--- | :--- |
| 1 | MeZO | 31GB |
| | FOCUS | 32GB |
| 16 | MeZO | 46GB |
| | FOCUS | **43GB** |

We believe the evidence above clarifies that the primary contribution of FOCUS provides a superior trade-off within the ZO paradigm, which operates in a fundamentally different memory class than any FO method.
We hope these extensive new results can further address your concerns and reaffirm the value and novelty of our work. Thank you again for your constructive review.