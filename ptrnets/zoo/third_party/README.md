# Third party resources
## CORnet
* Adapted from https://github.com/dicarlolab/CORnet
* Cite as follows:
```
@inproceedings{KubiliusSchrimpf2019CORnet,
abstract = {Deep convolutional artificial neural networks (ANNs) are the leading class of candidate models of the mechanisms of visual processing in the primate ventral stream. While initially inspired by brain anatomy, over the past years, these ANNs have evolved from a simple eight-layer architecture in AlexNet to extremely deep and branching architectures, demonstrating increasingly better object categorization performance, yet bringing into question how brain-like they still are. In particular, typical deep models from the machine learning community are often hard to map onto the brain's anatomy due to their vast number of layers and missing biologically-important connections, such as recurrence. Here we demonstrate that better anatomical alignment to the brain and high performance on machine learning as well as neuroscience measures do not have to be in contradiction. We developed CORnet-S, a shallow ANN with four anatomically mapped areas and recurrent connectivity, guided by Brain-Score, a new large-scale composite of neural and behavioral benchmarks for quantifying the functional fidelity of models of the primate ventral visual stream. Despite being significantly shallower than most models, CORnet-S is the top model on Brain-Score and outperforms similarly compact models on ImageNet. Moreover, our extensive analyses of CORnet-S circuitry variants reveal that recurrence is the main predictive factor of both Brain-Score and ImageNet top-1 performance. Finally, we report that the temporal evolution of the CORnet-S "IT" neural population resembles the actual monkey IT population dynamics. Taken together, these results establish CORnet-S, a compact, recurrent ANN, as the current best model of the primate ventral visual stream.},
archivePrefix = {arXiv},
arxivId = {1909.06161},
author = {Kubilius, Jonas and Schrimpf, Martin and Hong, Ha and Majaj, Najib J. and Rajalingham, Rishi and Issa, Elias B. and Kar, Kohitij and Bashivan, Pouya and Prescott-Roy, Jonathan and Schmidt, Kailyn and Nayebi, Aran and Bear, Daniel and Yamins, Daniel L. K. and DiCarlo, James J.},
booktitle = {Neural Information Processing Systems (NeurIPS)},
editor = {Wallach, H. and Larochelle, H. and Beygelzimer, A. and D'Alch{\'{e}}-Buc, F. and Fox, E. and Garnett, R.},
pages = {12785----12796},
publisher = {Curran Associates, Inc.},
title = {{Brain-Like Object Recognition with High-Performing Shallow Recurrent ANNs}},
url = {http://papers.nips.cc/paper/9441-brain-like-object-recognition-with-high-performing-shallow-recurrent-anns},
year = {2019}
}
```

## Taskonomy networks
* Adapted from visualpriors repo: https://github.com/alexsax/midlevel-reps/tree/visualpriors
* Cite repository:
```
@inproceedings{midLevelReps2018,
 title={Mid-Level Visual Representations Improve Generalization and Sample Efficiency for Learning Visuomotor Policies.},
 author={Alexander Sax and Jeffrey O. Zhang and Bradley Emi and Amir R. Zamir and Leonidas J. Guibas and Silvio Savarese and Jitendra Malik},
 year={2018},
}
```
* Cite paper:
```
@inproceedings{zamir2018taskonomy,
  title={Taskonomy: Disentangling task transfer learning},
  author={Zamir, Amir R and Sax, Alexander and Shen, William and Guibas, Leonidas J and Malik, Jitendra and Savarese, Silvio},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3712--3722},
  year={2018}
}
```