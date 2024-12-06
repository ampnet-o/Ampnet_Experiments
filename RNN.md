# Overview
Sanity testing a single layer residual RNN with `d_model=32`. ~3 minute dataset from GuitarML
- Trained on a CPU <3m of data, with early stopping and Error-Signal loss.

**hyperparams**
```
hp = {
    "batch_size": 40,         # ~4s / batch
    "ctx_len": 4410,          # 100ms sample
    "epochs": 1,
    "lr": 1e-4,
    "optim": "adam"
}
```

<!-- temp for now -->
![image](https://github.com/user-attachments/assets/be8e00ea-161e-42e2-8b24-3d2a83ff9882)

**Results**
```
N_params: 1121
FLOPs: 2463
Final loss: 0.0535
```

# arch
```
class BaselineRNN(nnx.Module):
  def __init__(self, width, rngs):
    self.rnn = RNN(SimpleCell(in_features=1,
                              hidden_features=width,
                              rngs=rngs
                              ))
    self.dense = nnx.Linear(in_features=width, out_features=1, rngs=rngs)
    self.d_model = width

  def __call__(self, inputs):
    x = self.rnn(inputs)
    return self.dense(x) + inputs
  
  def approx_flops(self):
    res = 0
    d = self.d_model

    # RNN block
    res += d**2 + (d-1)*d       # W @ h_{t-1} -- h: (d), W: (d x d)
    res += d                    # W @ x -- x: (1), W: (d)
    res += 2*d                  # bias + hidden + in
    res += 8*d                  # elementwise tanh

    # Linear block
    res += 2*d - 1              # W @ h
    res += d                    # bias

    return res
```
