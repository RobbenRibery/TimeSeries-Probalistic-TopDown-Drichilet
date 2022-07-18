  output = F.softmax(output)
/Users/ericliu/Library/Python/3.8/lib/python/site-packages/torch/autograd/__init__.py:173: UserWarning: Error detected in MmBackward0. Traceback of forward call that caused the error:
  File "/Users/ericliu/Desktop/ML-Side Project/TS Forecasting/Kaggle - M5 - Sales Forecating (accuracy)/src/proption_model.py", line 665, in <module>
    p_model.train(
  File "/Users/ericliu/Desktop/ML-Side Project/TS Forecasting/Kaggle - M5 - Sales Forecating (accuracy)/src/proption_model.py", line 476, in train
    model_output = self.linear_output(value)  # L, C, 1
  File "/Users/ericliu/Library/Python/3.8/lib/python/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/Users/ericliu/Desktop/ML-Side Project/TS Forecasting/Kaggle - M5 - Sales Forecating (accuracy)/src/proption_model.py", line 241, in forward
    output = self.linear(x)
  File "/Users/ericliu/Library/Python/3.8/lib/python/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/Users/ericliu/Library/Python/3.8/lib/python/site-packages/torch/nn/modules/linear.py", line 103, in forward
    return F.linear(input, self.weight, self.bias)
 (Triggered internally at  /Users/distiller/project/pytorch/torch/csrc/autograd/python_anomaly_mode.cpp:104.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
  0%|                                                                                                                                      | 0/10 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/Users/ericliu/Desktop/ML-Side Project/TS Forecasting/Kaggle - M5 - Sales Forecating (accuracy)/src/proption_model.py", line 665, in <module>
    p_model.train(
  File "/Users/ericliu/Desktop/ML-Side Project/TS Forecasting/Kaggle - M5 - Sales Forecating (accuracy)/src/proption_model.py", line 492, in train
    batch_loss.backward(retain_graph=True)
  File "/Users/ericliu/Library/Python/3.8/lib/python/site-packages/torch/_tensor.py", line 363, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
  File "/Users/ericliu/Library/Python/3.8/lib/python/site-packages/torch/autograd/__init__.py", line 173, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [64, 1]], which is output 0 of AsStridedBackward0, is at version 2; expected version 1 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!