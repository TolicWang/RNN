## 基于 peepholes LSTM 的ConvLSTM

由于`contrib.rnn.ConvLSTMCell`中对于`ConvLSTMCell`的实现本没有基于原作者的所应用的带有 "peepholes connection"的LSTM。因此，这里就照着葫芦画瓢，直接在原来的`contrib.rnn.ConvLSTMCell`的`call()`实现中上添加了`peepholes`这一步。<br>

添加的代码为：
```python
        w_ci = vs.get_variable(
            "w_ci", cell.shape, inputs.dtype)
        w_cf = vs.get_variable(
            "w_cf", cell.shape, inputs.dtype)
        w_co = vs.get_variable(
            "w_co", cell.shape, inputs.dtype)

        new_cell = math_ops.sigmoid(forget_gate + self._forget_bias + w_cf * cell) * cell
        new_cell += math_ops.sigmoid(input_gate + w_ci * cell) * math_ops.tanh(new_input)
        output = math_ops.tanh(new_cell) * math_ops.sigmoid(output_gate + w_co * new_cell)
```

引用时，将 `ConvLSTM`中的`BasicConvLSTM`导入即可：
```python
from ConvLSTM import BasicConvLSTM
```
用法同`ConvLSTMCell`一模一样！