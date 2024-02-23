# PGExplainer
This is a Tensorflow implementation of the paper: <b>Parameterized Explainer for Graph Neural Network</b>

https://arxiv.org/abs/2011.04573

<i>NeurIPS 2020</i>


<i>TPAMI 2024</i>

## Requirements
  * Python 3.6.8
  * tensorflow 2.0
  * networkx

# Pytorch Implementations
Now, PGExplainer is avilable at pytorch_geometric

https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/explain/algorithm/pg_explainer.py

Here are several re-implementations and reproduction reports from other groups.
Thanks very much these researchers for re-implementing PGExplainer to make it more easy to use!

1. [Re] Parameterized Explainer for Graph Neural Network 

https://zenodo.org/record/4834242/files/article.pdf


Code: 

https://github.com/LarsHoldijk/RE-ParameterizedExplainerForGraphNeuralNetworks


<b>Note that in this report, they adopt different GCN models with our implementation.</b>

2.  DIG

https://github.com/divelab/DIG/tree/main/dig/xgraph/PGExplainer


3. Reproducing: Parameterized Explainer for Graph NeuralNetwork

https://openreview.net/forum?id=tt04glo-VrT

Code: 

https://openreview.net/attachment?id=tt04glo-VrT&name=supplementary_material

4.  GitLab
https://git.gtapp.xyz/zhangying/pgexplainer


## Awesome Graph Explainability Papers

https://github.com/flyingdoog/awesome-graph-explainability-papers



## References
```
@article{luo2020parameterized,
  title={Parameterized Explainer for Graph Neural Network},
  author={Luo, Dongsheng and Cheng, Wei and Xu, Dongkuan and Yu, Wenchao and Zong, Bo and Chen, Haifeng and Zhang, Xiang},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
```
@article{luo2024towards,
  title={Towards Inductive and Efficient Explanations for Graph Neural Networks},
  author={Luo, Dongsheng and Zhao, Tianxiang and Cheng, Wei and Xu, Dongkuan and Han, Feng and Yu, Wenchao and Liu, Xiao and Chen, Haifeng and Zhang, Xiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```


