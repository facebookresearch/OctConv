How to use OctConv?

~~`from mxnet.gluon import nn`~~

`import octconv as nn`


Some examples for popular networks are given under `utils`.
You can run `plot_network.py` to plot the network architecture.

for example:
```
python plot_network --model mobilenet_v1_075
```

Note:
We use 'symbol api' for ablation study and 'gluon api' for other experiments.
