How to use OctConv?

~~`import mxnet.symbol as sym`~~

`import symbol_octconv as sym`


Some examples for popular networks are given under this folder.
You can run the scipt to plot the network architecture.

for example:
```
python symbol_resnetv2.py
```

It will show the total number of parameters and the computational graph. 
It will also generate the `*.symbol` file for calculating FLOPs.

Note:
We use 'symbol api' for ablation study and 'gluon api' for other experiments.
Depthwise OctConv is implemented but has not been tested under 'symbol api'.
