# MNIST Reimplementation Notes

This file captures the *essence* of the MNIST implementation in
`neural-net-toys` (written in Rust, ~2023) so it can be hand-ported to Mojo.
It assumes Mojo behaves roughly like CUDA for the relevant concepts (kernel
launches, shared memory, GEMM, elementwise ops). The goal was to write down
*all the math* in one place — the math was the most time-consuming part to get
right originally, largely because of backpropagation.

The original source files this is derived from:

- `src/main.rs` — CLI, data loading, network assembly
- `src/network.rs` — `Network` = flat list of `Layer`s; forward/backward orchestration
- `src/layers.rs` — every layer's forward/backward math
- `src/im2col.rs` — the conv-as-GEMM trick
- `src/optimizers.rs` — GD and AdamW
- `src/initialization.rs` — Xavier / He weight init
- `src/training.rs` — the training schedule / epochs / batches / early stop

---

## 1. Layout conventions (read this first)

### 1.1 Everything is a 2-D batch matrix `[examples, features]`

Every layer, forward and backward, takes an `ArrayView2<f32>` (shape
`(examples, features)`) and returns the same. This is the single most
important convention in the whole codebase — port it to Mojo *religiously*.

- `examples` is the **row** axis (index 0).
- `features` is the **column** axis (index 1).

### 1.2 MNIST data shapes (from `main.rs`)

| thing | shape | notes |
|---|---|---|
| images | `[50000, 784]` | 28×28 flattened; each pixel normalized `/255.0` |
| train | 50000 | |
| validation | 10000 | downloaded but **not used** in training |
| test | 10000 | |
| labels | `[50000, 10]` | one-hot |

### 1.3 Layer outputs are always of the form `examples × features`, but "features"
means different things per layer

- Fully connected / activation / dropout: features = flat neuron count.
- Conv / pool: features = `channels × H × W`, stored **row-major with channel
  outermost** (`channel * H * W + row * W + col`). This flattened layout is
  identical for conv and pool so their outputs chain together without a reshape.

When a conv/pool layer needs 4-D `(batch, channels, height, width)` it
reshapes **per example** by slicing a single row. Practically this means your
kernels can be *per-example* (one thread-block per sample) or *batched* — the
choice is yours, but keep `examples × features` as the interface at every
layer boundary.

---

## 2. Network assembly (what gets built)

`Network` is just `Vec<Layer>` and is assembled imperatively. **Activation
functions are pushed as their own separate layer objects**, not fused into the
linear layers. With the MNIST defaults (`--activation tanh`, 2 conv layers,
kernel 5, one hidden layer of width 128) the list is:

```
conv(1 → 8,  k=5)   → tanh  → pool(2×2)  → dropout
conv(8 → 16, k=5)   → tanh  → pool(2×2)  → dropout
fc(784? → 128)      → tanh  → dropout
fc(→ 10)            → softmax
```

Notes:

- The first `fc` input width is whatever the previous flatten produces
  (`16 × 7 × 7` if pooling worked as intended — see the bug in §9).
- `conv(1 → 8)`: first conv goes 1 channel → 8. Subsequent convs double the
  channels (`channels_out = channels_in * 2`).
- The final `fc(→ 10)` is always softmax (categorical — always one layer must
  represent the class index).
- Dropout default `keep_probability = 1.0 - 0.5 = 0.5`.

### 2.1 `Layer` interface (the whole design secret)

```rust
trait Layer {
    fn forward(&self, input: &ArrayView2<f32>) -> Array2<f32>;
    fn loss(&self, output, target) -> Array1<f32>;         // per-example loss
    fn dloss_doutput(&self, output, target) -> Array2<f32>; // ∂L/∂output
    fn backward(&mut self, input, dloss_doutput) -> Array2<f32>;
    //   ↑ returns ∂L/∂input AND stores ∂L/∂params internally
    fn layer_state_mut(&mut self) -> Vec<LayerStateMut>;    // (params, grad)
    // plus start/end_training_step hooks used only by dropout
}
```

A layer only knows its own math and exposes `(params, grad)` to a generic
optimizer. The optimizer never sees layer internals. This is the abstraction
to reproduce: **forward + backward + a generic way to hand the flattenable
parameter/gradient buffers to the optimizer.**

---

## 3. The training schedule (`training.rs::train`)

Per run:

1. Pick an optimizer (GD or AdamW; **AdamW is the default**).
2. For each **epoch** (default up to 2000, but early-stopping kills it):
   a. Compute number of batches: `batch_count = ceil(train_count / batch_size)`,
      `batch_size = 32`.
   b. **Shuffle the batch indices** (only the *order of batches* is randomized,
      not individual samples).
   c. For each batch:
      - `network.compute_gradients(inputs, targets)` — forward + backward.
      - `optimizer.optimize(network)` — update params.
      - compute batch loss; **abort if non-finite**.
   d. Full pass over the **test set** (10k samples, one at a time = batch 1)
      to compute test loss & accuracy.
   e. Record epoch stats, redraw UI, write JSONL history.
   f. **Early stop**: if the last `patience` (5) epochs all have test accuracy
      strictly below the best so far → stop.
3. The **best model** (highest test accuracy) is cloned and saved on the fly.

The clean separation **compute_gradients then optimize** is deliberate
(commit "Split gradient computation from parameter updates") — exactly the
pattern you want on a GPU: run gradient kernels, then a separate update
kernel.

---

## 4. Forward/backward orchestration (`network.rs::compute_gradients`)

```
1. start_training_step() on every layer     // dropout picks its mask
2. Forward, caching EVERY layer's output:
     out[0] = input
     out[i+1] = layer[i].forward(out[i])
   output = out[-1]
3. dloss_doutput = last_layer().dloss_doutput(output, target)
4. Backward in reverse over layers:
     for i in reverse:
        dloss_doutput = layer[i].backward(out[i], dloss_doutput)
        // each call returns ∂L/∂input AND stores ∂L/∂params
5. end_training_step() on every layer       // dropout resets mask to 1.0
```

Key point: **the forward pass materializes every layer output**, and backward
consumes those cached activations in reverse order. Your port must hold all
layer activations (or recompute them cleverly). There is no memory
optimization here — it's plain vanilla backprop with materialized caches.

---

## 5. Loss and the softmax trick

The default `Layer::loss` / `Layer::dloss_doutput` is **mean squared error**:

$$
\text{loss} = \frac{1}{n_{\text{feat}}}\sum_{o}\bigl(\hat{y}_o - y_o\bigr)^2
\qquad
\frac{\partial L}{\partial \hat{y}_o} = \frac{2}{n_{\text{feat}}}\,(\hat{y}_o - y_o)
$$

where `n_feat` = number of features (columns). `SoftmaxLayer` **overrides**
both:

$$
\text{loss} = -\sum_{o} y_o \ln \hat{y}_o \qquad\text{(per-example categorical cross-entropy)}
$$

$$
\frac{\partial L}{\partial \hat{y}_o} = \hat{y}_o - y_o
$$

The second line is the famous **softmax + cross-entropy collapse**: even
though softmax has a dense Jacobian, the composition with cross-entropy
reduces to the simple subtraction `ŷ − y`. So `SoftmaxLayer::backward` just
passes that gradient through unchanged (no softmax Jacobian applied at all).

> **Flag:** this backward shortcut is **only** valid because the loss is
> cross-entropy. If you ever swap the loss, you must reintroduce the full
> softmax Jacobian. Keep `loss` and `dloss_doutput` together as one unit.

---

## 6. Layer math

### 6.1 FullyConnectedLayer
```
weights W : (I, O)      biases b : (O,)
x : (B, I)              z : (B, O)
```

**Forward**
$$
z_{b,o} = \sum_i x_{b,i}\, W_{i,o} + b_o
\qquad\Longleftrightarrow\qquad
Z = X W + b \quad\text{(broadcast b over rows)}
$$

**Backward** — given $\frac{\partial L}{\partial Z}$ (shape `(B,O)`):

$$
\frac{\partial L}{\partial b_o} =
\sum_{b} \frac{\partial L}{\partial Z_{b,o}}
\qquad\Longleftrightarrow\qquad
\frac{\partial L}{\partial b} = \text{colsum over batch of } \frac{\partial L}{\partial Z}
$$

$$
\frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial Z}
\qquad (I,B) \cdot (B,O) \to (I,O)
$$

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial Z}\, W^\top
\qquad (B,O) \cdot (O,I) \to (B,I)
$$

(The code implements $X^T \frac{\partial L}{\partial Z}$ directly — this is the
sum-over-batch of the outer products, see §8.1.)

**Derivation (chain rule).** Each output is a weighted sum, so every parameter is on the path from *every* input to its output:

- $z_{b,o}$ depends on $b_o$ through $+b_o$ with slope 1 ⇒ $\frac{\partial L}{\partial b_o} = \sum_b \frac{\partial L}{\partial z_{b,o}}$ (sum over the batch index $b$, which does *not* appear in $b_o$).
- $z_{b,o}$ depends on $W_{i,o}$ through $x_{b,i}\cdot W_{i,o}$ ⇒ $\frac{\partial L}{\partial W_{i,o}} = \sum_b x_{b,i}\frac{\partial L}{\partial z_{b,o}}$, which is exactly the $(i,o)$ entry of $X^T\frac{\partial L}{\partial Z}$.
- $z_{b,o}$ depends on $x_{b,i}$ through $W_{i,o}$ ⇒ $\frac{\partial L}{\partial x_{b,i}} = \sum_o W_{i,o}\frac{\partial L}{\partial z_{b,o}}$, which is the $(b,i)$ entry of $\frac{\partial L}{\partial Z}W^T$.

The sum-over-batch in the first two (rather than a mean) is the convention this code uses throughout (§8.1).

### 6.2 TanhLayer
**Forward:** $z = \tanh(x)$

**Derivation.** Pure chain rule with no parameter sums: the activation maps each input element independently, so $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z}\cdot\frac{dz}{dx}$ and $\frac{dz}{dx} = \text{sech}^2(x) = 1 - \tanh^2 x$ (standard identity). This could be written `1 − z²` if forward stored its output $z$, but the code rebuilds `tanh(x)` from the cached *input* instead — same answer, a bit more compute, less memory.

**Backward:**
$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot (1 - \tanh^2 x)
$$

Note: derivative is recomputed from the cached input
(`1 - tanh(x)²`), not stored — a memory-for-compute tradeoff that is fine to
keep.

### 6.3 LeakyReLU
**Forward:** $z = x$ if $x>0$, else $\alpha x$ (default $\alpha = 0.01$)

**Backward:**
$$
\frac{\partial L}{\partial x} =
\frac{\partial L}{\partial z} \cdot \begin{cases} 1 & x \ge 0 \\ \alpha & x < 0 \end{cases}
$$

**Derivation.** Same as tanh: elementwise local slope $\frac{dz}{dx}$ is $1$ where $x>0$ (identity) and $\alpha$ where $x<0$ (scaled identity). At the kink $x=0$ the slope is not uniquely defined; taking the $+\epsilon$ limit (slope $1$) is the standard, fastest choice and is what this code does.

### 6.4 SoftmaxLayer
**Forward:** for each example, over outputs $o$:
$$
\hat{y}_o = \frac{e^{x_o}}{\sum_{c} e^{x_c}}
$$

**Derivation.** The generic softmax Jacobian is dense: $\frac{\partial\hat y_o}{\partial x_{o'}} = \hat y_o(\delta_{oo'} - \hat y_{o'})$. Combining with CE $L=-\sum_c y_c\ln\hat y_c$, whose per-term derivative is $\frac{\partial L}{\partial \hat y_o} = \frac{-y_o}{\hat y_o}$, the chain rule gives
$$\frac{\partial L}{\partial x_{o'}} = \sum_o \frac{-y_o}{\hat y_o}\cdot\hat y_o(\delta_{oo'}-\hat y_{o'}) = -y_{o'} + \hat y_{o'}\underbrace{\sum_o y_o}_{=1} = \hat y_{o'} - y_{o'}.$$
The $\sum_o y_o=1$ step (one-hot labels) is what makes everything collapse to $\hat y - y$. This is *why* softmax+CE is fused in practice: the naive alternative (build the $O\times O$ Jacobian then multiply) is wasteful and numerically fiddly.

**Backward / loss:** see §5. No softmax Jacobian is applied.

### 6.5 ConvLayer — the big one
```
weights W : (C_out, C_in, k, k)      biases b : (C_out)
input  x : (C_in, H, W)  [per example]
output z : (C_out, H, W)  [same spatial size]
padding p = (k-1)/2   (SAME padding; output H, W unchanged; k must be odd)
```

**Forward:**
$$
z_{c_o,\,h,\,w} =
\sum_{c_i}\,\sum_{k_h}\,\sum_{k_w}\;
x_{c_i,\,h+k_h,\,w+k_w}\, W_{c_o,\,c_i,\,k_h,\,k_w}
\;+\; b_{c_o}
$$

The input is zero-padded by `p` on all sides first, so the index
`h + k_h` never goes out of bounds (no bounds-checking in the hot loop).

In the code this is computed with **im2col** (see §7):
1. Pad input.
2. Extract, for each output pixel `(h,w)`, the patch of length
   `C_in·k²` and gather into a `patches` matrix `[H·W, C_in·k²]`.
3. Reshape weights to `[C_out, C_in·k²]` and compute
   `Z = W_matrix · patchesᵀ` (a **GEMM**), then add the broadcast bias.

**Derivation.** Convolution is linear in all three inputs (weights, input, bias), so each backward term is a fresh convolution-like sum — you only need to ask *"which variables does this output depend on, and with what coefficient?"* then chain-rule the one free index.

- $z_{c_o,h,w}$ depends on $W_{c_o,c_i,k_h,k_w}$ with coefficient the input at the sliding position $x_{c_i,h+k_h,w+k_w}$ ⇒ summing that coefficient over every $(h,w)$ where it appears gives the $\partial L/\partial W$ formula below.
- $z_{c_o,h,w}$ depends on $b_{c_o}$ with coefficient $1$ ⇒ $\partial L/\partial b$ is the sum of $\partial L/\partial Z$ over all spatial/batch positions of that channel.
- $z_{c_o,h,w}$ depends on $x_{c_i,h+k_h,w+k_w}$ with coefficient $W_{c_o,c_i,k_h,k_w}$ ⇒ to get $\partial L/\partial x$ at a *fixed* cell $(c_i,h,w)$ you sum over all $c_o$ and kernel offsets that reach it, flipping the kernel index to $k-k_h-1$ (the 180° rotation). The reversal appears because the "input→output" offset $+k_h$ becomes "output→input" offset $-k_h$, and $k-k_h-1$ wraps it back into $[0,k)$.

**Backward** — given $\frac{\partial L}{\partial Z}$ (shape `(C_out,H,W)` per
example, summed/accumulated over the batch below):

**∂L/∂weights:**
$$
\frac{\partial L}{\partial W_{c_o,c_i,k_h,k_w}} =
\sum_{\text{batch}}\, \sum_{h}\,\sum_{w}\;
\underbrace{x_{c_i,\,h+k_h,\,w+k_w}}_{\text{padded input}}\;\cdot\;
\frac{\partial L}{\partial Z_{c_o,\,h,\,w}}
$$

**∂L/∂biases:**
$$
\frac{\partial L}{\partial b_{c_o}} =
\sum_{\text{batch}}\, \sum_{h}\,\sum_{w}\;\frac{\partial L}{\partial Z_{c_o,\,h,\,w}}
$$

**∂L/∂input** (gradient w.r.t. the *unpadded* input `x`):
$$
\frac{\partial L}{\partial x_{c_i,\,h,\,w}} =
\sum_{c_o}\,\sum_{k_h}\,\sum_{k_w}\;
\frac{\partial L}{\partial Z_{c_o,\,h+k_h,\,w+k_w}}\;
W_{c_o,\,c_i,\,k-k_h-1,\,k-k_w-1}
$$

i.e. the gradient is a **full correlation of ∂L/∂Z with the 180°-rotated
kernel** (channels of input and output swapped, kernel flipped in both spatial
axes). The code implements this by building a `flip_kernel` (see §7) and
reusing the exact same im2col convolution routine — so conv *forward* and the
conv *input-gradient* share one kernel implementation.

Note the `∂L/∂weights` accumulation is a 7-deep loop in the code
(`example, c_out, h, w, c_in, k_h, k_w`). It *could* be written as
`(∂L/∂Z)ᵀ · patches` — another GEMM. That's a natural optimization for a Mojo
port and is mathematically identical.

### 6.6 PoolLayer (max pool)
```
input  x : (C, H, W)  →  output z : (C, outH, outW),  kernel k, stride s
outH = ceil(H / s),  outW = ceil(W / s)
```
For MNIST defaults: k = 2, s = 2 → spatial dims halve (28 → 14 → 7).

**Forward:**
$$
z_{c,\,h_o,\,w_o} = \max_{\substack{k_h \in [0,k)\\ k_w \in [0,k)}}\;
x_{c,\; h_o s + k_h,\; w_o s + k_w}
$$

Input is padded on the right/bottom with `−∞` so max still works at borders;
the padding region never wins the max.

**Backward** — max pooling has **no parameters**. The gradient is routed to
the *argmax* position:
$$
\frac{\partial L}{\partial x_{c,\,h_i,\,w_i}} =
\begin{cases}
\displaystyle\sum_{(h_o,w_o) \in \text{positions whose window's max is at }(h_i,w_i)}
\frac{\partial L}{\partial z_{c,\,h_o,\,w_o}} & \text{if } (h_i,w_i) \text{ is the argmax of some window}\\[2mm]
0 & \text{otherwise}
\end{cases}
$$

**Derivation.** Max selects a *subset* of its inputs: the discarded ones are irrelevant to $L$ (zero derivative), and the kept one passes through with slope 1, so its derivative is exactly the upstream gradient. Argmax is not differentiable (ties are measure-zero), so picking *a* winner is fine. This routes $\frac{\partial L}{\partial z_{c,h_o,w_o}}$ — the local chain-rule step — matching `backward`'s contract of returning $\partial L/\partial x$.

### 6.7 DropoutLayer
- has **no trainable parameters**
- the mask is a per-feature vector `m` of length `width`

Each training step (`start_training_step`):
1. `keep = keep_probability · width` (rounded).
2. Set the first `keep` entries to the **inverted scale** `1 / keep_probability`,
   the rest to `0`.
3. Shuffle the mask.

**Forward:** $\;\hat{x} = x \cdot m$

**Derivation.** With the mask $m$ frozen for a step (chosen once, then fixed for that forward+backward), dropout is just an elementwise multiplication by constants, so the chain rule is trivial: $\frac{d\hat x}{dx}=m$, hence $\frac{\partial L}{\partial x} = m\cdot\frac{\partial L}{\partial \hat x}$. Because only *kept* units have $m\ne 0$, gradients to dropped units are zeroed automatically — no separate masking of backward needed. The $1/\text{keep}$ scaling on kept units makes the *expected* value of $\hat x$ equal to $x$ (inverted dropout), so no scaling is needed at inference and test results stay unbiased.

**Backward:** $\;\dfrac{\partial L}{\partial x} = \dfrac{\partial L}{\partial \hat{x}} \cdot m$

**After the step** (`end_training_step`): mask = all `1.0`, so **inference is
unaffected** — no scaling at inference time, because the `1/keep` scaling was
baked into the retained entries. This is *inverted dropout*; it's the
convention to keep.

> **Flag:** the in-code comment says "set exactly `keep_probability*width` values
> to 1.0" but the code actually sets them to `1/keep_probability`. The **code is
> correct** (that's the inverted-dropout scaling); the comment is just stale.

---

## 7. im2col (the conv-as-GEMM trick, `im2col.rs`)

The whole point: **turn convolution into matrix multiplication**, exploiting
the BLAS/GPU GEMM machinery. This is exactly what you want in Mojo.

Conceptual shapes (per example):
```
img shape               : (C_in, H, W)
patches shape           : (H·W, C_in·k·k)     ← "unroll" the image
kernel reshaped to      : (C_out, C_in·k·k)
out shape               : (C_out, H·W)  →  reshape to (C_out, H, W)
out = kernel_matrix · patchesᵀ
```

Padding: `p = k/2` (integer division; k odd so = (k−1)/2). The implementation
pre-allocates all intermediate storage to avoid allocations in hot loops.

**Backward input gradient reuses im2col.** To compute `∂L/∂x`, build a
`flip_kernel`:

```
flipped[c_in, c_out, kh, kw] = W[c_out, c_in, k-1-kh, k-1-kw]
```

(channels swapped *and* kernel rotated 180°), then run the *same* im2col conv2d
with `∂L/∂Z` as the "image" and `flipped` as the "kernel":

```
dloss_dimg = im2col_conv2d(dloss_dZ, flipped_kernel)
```

---

## 8. The optimizer (AdamW, `optimizers.rs`)

The optimizer iterates `network.network_state_mut()` — a flat list of
`(params, grad)` typed arrays pulled out of every layer, in order. Conceptually
this is a **flat parameter buffer** (plus a matching flat gradient buffer).

### 8.1 Gradient scaling: summed, not averaged — READ THIS

Backprop in this code **sums the gradients over the batch** rather than
averaging them. Concretely, in `FullyConnectedLayer::backward`:

- `∂L/∂bias = ∑_batch ∂L/∂Z`  (a sum)
- `∂L/∂W = Xᵀ · ∂L/∂Z`  (matmul — sums over the batch implicitly)

And `SoftmaxLayer::dloss_doutput = ŷ − y` has **no** `1/batch` (or `1/n_feat`)
factor. So the gradient magnitude scales with batch size (32).

**Why this mostly works:** AdamW normalizes each parameter update by
`√v̂ + ε`, which is a moving variance of the (scaled) gradient, so the
sum-vs-mean choice mostly cancels out into an effective learning-rate constant.

**Why it's worth flagging:**
- The *reported* train loss is the mean of per-example cross-entropy over the
  whole epoch (`train_loss / train_count`), but the *actual* gradient being
  minimized is a plain sum over the batch with no normalization. So the number
  you see in the loss curve and the objective the optimizer descends are *not*
  the same scaled quantity. Benign when Adam normalizes, but if you wanted
  deterministic/bit-comparable results, you'd need to nail this down.
- When you port, **decide deliberately** whether gradients are summed or meaned
  over the batch, and be consistent — don't mix conventions between layers.
  The reference implementation is "summed" throughout.

### 8.2 Plain gradient descent
$$
\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}
$$

### 8.3 AdamW (default)

**Derivation (conceptual).** The gradient $g_t$ is treated as a *random* sample of the true gradient, so Adam maintains a running estimate of its **expected value** ($m$, first moment) and **expected squared magnitude** ($v$, second raw moment). Two exponentially-weighted moving averages do this:

- $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$ — puts exponentially more weight on recent gradients, smoothing the noisy per-batch gradient into a stable direction.
- $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ — tracks typical *magnitude*, which becomes the per-parameter step-size denominator, giving every parameter a self-tuned step: large-gradient params step small, small-gradient params step large (adaptive rate, like Adagrad/RMSProp but with EWMA).
- **Bias correction.** Both $m$ and $v$ start at $0$, so early estimates are biased low. An EWMA with decay $\beta$ after $t$ steps has total weight $1-\beta^t$; dividing by $(1-\beta^t)$ removes the bias exactly. $\beta^t\to 0$, so the correction fades to 1 as $t$ grows.
- $\hat m_t / (\sqrt{\hat v_t}+\epsilon)$ — a normalized, roughly unit-variance *direction* per parameter, times the global rate $\eta$. The $+\epsilon$ (e.g. `1e-8`) is a guard against dividing by ~0.
- **Weight decay `−λθ` is separate** (that's the *W*). In plain Adam+L2 the decay is folded into $g_t$ and thus into $m,v$, coupling it to the adaptive scaling; AdamW applies it directly to $\theta$ so it behaves like true L2/weight decay regardless of the per-param scaling. Loosely: Adam picks *steps*, weight decay constantly *pulls weights toward 0*.

Per-parameter state: `m` (first moment / momentum) and `v` (second raw moment /
variance), both initialized to **0**.

Step `t` (increment `t` on each optimize call; `saturating_add` so overflow
just clamps and the bias corrections → 1):

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \\
\hat m_t &= \frac{m_t}{1-\beta_1^{\,t}} \qquad
\hat v_t = \frac{v_t}{1-\beta_2^{\,t}} \\
\theta_t &= \theta_{t-1}
- \eta\,
\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
- \lambda\,\theta_{t-1}
\end{aligned}
$$

- $\eta$ (learning rate): default from CLI `--learning-rate` = `0.01`
- $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon$ = `--adamw-epsilon` = `1e-8`
- $\lambda$ (weight-decay) = `--l2-regularization`
- **The `−λ·θ` term is *decoupled* from the adaptive step** — that is what
  makes it *AdamW* rather than Adam+L2. It's applied to the raw parameter, not
  folded into `g`.

> **Flag:** the `AdamWOptimizerBuilder` default is `lambda = 0.01`, but `train()`
> overrides it with `opt.l2_regularization`, whose CLI default is **0.0**. So
> unless you pass `--l2-regularization`, weight decay is actually **off** even
> though the builder default suggests 0.01. The apparent discrepancy is
> intentional but easy to misread.

**Porting note:** `m` and `v` are (conceptually) arrays the same shape as the
flat parameter buffer, one pair per layer-state. The update is a set of
elementwise kernels over the flat buffer (multiply, add, sqrt, div, subtract).
Trivial to parallelize once the flat `[params | grad | m | v]` layout exists.

---

## 9. Weight initialization (`initialization.rs`)

| Activation feeding into the layer | init | formula |
|---|---|---|
| tanh / softmax | **Xavier** | uniform over $[-1/\sqrt{n}, 1/\sqrt{n}]$ → width $= 2/\sqrt{n}$ |
| leaky ReLU | **He** | $\mathcal{N}(0,\; 2/n)$ (std $= \sqrt{2/n}$) |

where $n$ = number of inputs to the layer (fan-in):

- Fully connected: `n = input_size`
- Conv: `n = C_in · k · k` (fan-in per output channel), then reshaped to
  `(C_out, C_in, k, k)`

Note this Xavier uses **only fan-in** (not the `fan_in + fan_out` variant) and
a uniform distribution — a legitimate simplification of Glorot; worth keeping
as-is for reproducibility. **Biases are always initialized to zero.**

---

## 10. Bugs / flags found while writing this up

1. **Spatial bookkeeping bug in `main.rs` (`train_mnist`).** After each
   conv+pool block the code updates the tracked image size with
   ```
   layer_img_height /= opt.kernel_size;   // divides by 5 (kernel), but ...
   layer_img_width  /= opt.kernel_size;   // ... pooling actually halves ÷ 2
   ```
   The pool layer reduces spatial dims by `stride` (2), not by `kernel_size`
   (5). So the tracked size for the *next* conv layer is wrong
   (28 → 5 → 0 with k=5, vs the real 28 → 14 → 7). Because `ConvLayer::new`
   stores that tracked size and uses it to allocate/reshape, the multi-conv
   path would mis-shape and fail. **When porting, track the real `ceil(H/s)`
   spatial size produced by each pool layer and ignore this line.** (The pool
   layer's own math is correct; only asynchronous bookkeeping is broken.)

2. **Summed (not meaned) gradients + mixed loss scaling** (§8.1). The gradient
   backprop is a sum over the batch, while the reported loss is a mean over
   the epoch. Works under AdamW, but pick one convention in the port and be
   consistent.

3. **Softmax backward shortcut is loss-dependent** (§5/§6.4). Valid only
   because the loss is cross-entropy.

4. **Dropout mask value: code is right, comment is stale** (§6.7). The
   retained units are set to `1/keep_probability`, not `1.0`.

5. **AdamW λ default mismatch** (§8.3). Builder default 0.01 gets overridden by
   CLI default 0.0 → weight decay effectively off by default.

6. While reading, note the conv `∂L/∂W` is a 7-deep loop; it can be a GEMM
   (`(∂L/∂Z)ᵀ · patches`). Not a bug — a free optimization opportunity for the
   port.

---

## 11. The essence in one paragraph

One 2-D batch matrix `[examples, features]` flows through a flat list of
layers, each exposing forward/backward and hands its flat `(params, grad)`
buffers to a generic AdamW optimizer. Softmax+CE at the head collapses to a
single `ŷ − y` gradient; backward propagates `∂L/∂input` in reverse while each
layer stashes `∂L/∂params` (summed over the batch). Convolution is im2col →
GEMM forward and a flipped-kernel im2col GEMM backward; max-pool routes
gradients to argmax positions. The schedule is: shuffle batches → per batch
(forward, backward, update) → per epoch full test pass → early-stop when test
accuracy stalls for `patience` epochs, cloning the best model along the way.
