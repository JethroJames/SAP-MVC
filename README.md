# SAP-MVC

Trustworthy multi-view classification (TMVC) addresses the challenge of achieving reliable decision-making in complex scenarios where multi-source information is heterogeneous, inconsistent, or even conflicting. Existing TMVC approaches predominantly rely on globally dense neighbor relationships to model intra-view dependencies, leading to high computational costs and an inability to directly ensure consistency across inter-view relationships. Furthermore, these methods typically aggregate evidence from different views through manually assigned weights, lacking guarantees that the learned multi-view neighbor structures are consistent within the class space, thus undermining the trustworthiness of classification outcomes. To overcome these limitations, we propose a novel TMVC framework that introduces prototypes to represent the neighbor structures of each view. By simplifying the learning of intra-view neighbor relations and enabling dynamic alignment of intra- and inter-view structures within each minibatch, our approach facilitates more efficient and consistent discovery of cross-view consensus. Extensive experiments on multiple public multi-view datasets demonstrate that our method achieves competitive downstream performance and robustness compared to prevalent TMVC methods. Our code is here.

# How to set conflict datasets?

Add code lines to the text:

dataset.postprocessing(test_index, addNoise=True, sigma=0.5, ratio_noise=0.1, addConflict=True, ratio_conflict=0.4)
