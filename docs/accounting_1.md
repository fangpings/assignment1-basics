# Problem (transformer_accounting): Transformer LM resource accounting

## 1

Consider GPT-2 XL, which has the following configuration:

```
vocab_size : 50,257 
context_length : 1,024
num_layers : 48 
d_model : 1,600
num_heads : 25
d_ff : 6,400
```

Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

Answer

```
v: vocab_size
h: d_model
c: context_length
l: num_layers
d_ff = 4h

Embedding: v * h = 50,257 * 1,600 = 80.41m

LM Head: v * h = 50,257 * 1,600 = 80.41m

Attn:
    attn_proj: 3 * h * h = 3 * 1,600 * 1,600 = 7.68m
    output_proj: h * h = 1,600 * 1,600 = 2.56m
    total = 10.24m

FFN:
    w1=w2=w2: h * d_ff = 1600 * 6400 = 10.24m
    total: 30.72m

Block:
    total: 40.96m

Total: num_layers * block + embedding + lm_head 
    = 2vh + 16lh^2
    = 48 * 40.96 + 2 * 80.41 = 2.12B

RMS Norm too small and omitted

If each parameter is stored in fp32 (4 bytes), then it's 8gb
```
Note that model size is calculated by 
$$
2vh + 16lh^2
$$
And it's actually dominant by hidden size h


Note: the result is slightly different from the original [gpt2-xl](https://huggingface.co/openai-community/gpt2-xl) that has 1.5B parameters because our architecture changed. The main difference on parameter is in FFN, where the original one only has 2 matrices but swiglu has 3, so for a single ffn layer we have 10m more parameters. Also original GPT2 reuses token embedding for lm head, and that saves 80m parameters. Original GPT2 also does not use Rope but a learnable PE layer instead, but that is small in terms of the number of parameters.

## 2

Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.

Given $A ∈ R ^{m×n}$ and $B ∈ R ^{n×p}$, the matrix-matrix product AB requires 2mnp FLOPs.

Answer

```
Embedding: 2 * c * v * h = 164.7B
LM Head: 164B

Attn:
    attn_proj: 2 * c * 3 * h * h = 15.7B
    scaled_dot_product_attention:
        qk: 2 * c * h * c = 3.3B
        v: 2 * c * h * c = 3.3B
    output_proj: 2 * c * h * h = 5.2B
    total: 27.5B

FFN:
    w1/w2/w3: 2 * c * h * d_ff = 20.9B
    total: 62.7B

Block: 90.2B

Total: 
    2cvh + l(8ch^2 + 4c^2h + 8ch^2)
    = 2cvh + l(16ch^2 + 4c^2h)
    = 164B *2 + 90.2B * 25
    = 2.58T

All pointwise operations are omitted (rope, softmax and activation)
```

Note that computation complexity is calculated by
$$
2cvh + l(16ch^2 + 4c^2h)
$$
Usually c is larger than h by a lot (for example a lot of model supports 128k c), so it's mainly dominant by context length


