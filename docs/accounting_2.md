Let us compute how much memory and compute running AdamW requires. Assume we are using float32 for every tensor.

## Space

How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of the batch_size and the model hyperparameters (vocab_size, context_length, num_layers, d_model, num_heads). Assume d_ff = 4 × d_model.

For simplicity, when calculating memory usage of activations, consider only the following components:

* Transformer block

  * RMSNorm(s)
  * Multi-head self-attention sublayer: QKV projections, Q⊤K matrix multiply, softmax, weighted sum of values, output projection.
  * Position-wise feed-forward: W 1matrix multiply, SiLU, W 2matrix multiply
* final RMSNorm
* output embedding
* cross-entropy on logits

```
vocab_size = v
context_length = s
num_layers = l
d_model = h
batch_size = b
num_heads = n
```

### Parameters

```
Embedding: vh
LM Head: vh

Layer:
    Attn: 4h^2
    FFN: 12h^2
    Total: 16h^2

Total: 2vh + 16lh^2
```

### Gradient

Same space as parameters

### Optimizer State

Twice the space as parameters

### Activations

```
Block:
    Norm1: bsh
    Attn:
        qkv proj: 3bsh
        qk multiply: bns^2
        softmax: bns^2
        weighted sum of values: bsh
        output projection: bsh
    Norm2: bsh
    FFN: 
        w1: 4bsh
        silu: 4bsh
        w2: bsh
Total: 16bsh+2bns^2

Final Norm: bsh
Output Embedding: bsv
Cross Entropy:  bsv

Total: (16bsh+2bns^2)l + bsh + 2bsv
```

### Total

```
4*(2vh + 16lh^2) + (16bsh+2bns^2)l + bsh + 2bsv
```

### GPT2-XL

```
v = 50257
s = 1024
l = 48
h = 1600
n = 25

constant memory: 4 * 4 * (2vh + 16lh^2) = 31.7GB
floating memory: b * 4 * ((16sh+2ns^2)l + sh + 2sv) = 14.4G * b

if you have 80G, then the max batch size you can have is 3
```

## Computation

P: number of parameters
B: batch size
S: sequence length

All flops are calculated for 1 step
### Forward

flops_foward = 2 * P * B * S

This approximation works because nearly every parameter in the model (especially in the large weight matrices) performs a multiply-accumulate operation for each token. The factor of 2 accounts for the two operations in a multiply-accumulate (1 multiplication + 1 addition).

### Backward

flops_foward = 4 * P * B *S

The backward pass takes roughly twice the computation of the forward pass. This is because for every matrix multiplication in the forward pass (e.g., Y=X⋅W), the backward pass must perform two matrix multiplications: one to find the gradient with respect to the weights (dW) and another to find the gradient with respect to the input (dX)

### Optimizer

flops_op = 16 * P

Optimizer calculation is actually negligible since it's invariant to number of input tokens, but
only related to steps

### GPT2-XL

GPT2-XL has 2.13B parameters (our version)

```
total flops: 2.13B * 400k * 1024 * 1024 * 6 = 5.36 * 10^9 T
singel A100 flops / day = 19.5 * 0.5 * 86400 = 8.4 * 10^5 T

number of days: 6380 days
```