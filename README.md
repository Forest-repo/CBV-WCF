# When Complementarity Meets Consistency: Weighted Collaboration Fusion Constrained by Consistency Between Views for Multi-View Urban Scene Classification
this is "CBV-WCF" paper
# Introduction
1. A novel Coupled Parallel Architecture (CPA) is proposed to implement end-to-end inference when there are some missing views in the input sample pairs. This greatly increases its flexibility and convenience, making it widely used in multi-view situations.
2. A Weighted Collaboration Fusion (WCF) module is proposed to fully explore and effectively fuse the potential complementary information between different views. WCF adopts the fusion strategy of “each takes what he needs”, which not only guarantees the independence and integrity of feature extraction from original views, but also fully and adaptively fuses complementary features required by views, significantly improving the performance of fusion classification in scenes with large feature differences between views.
3. An Consistency Between Views (CBV) module is proposed to extract cross-view consistency information. The network can give feature consistency precedence over semantic consistency using CBV. This can somewhat lessen the challenges posed by inter-class similarity and intra-class diversity, enhancing the model's performance in classification and its portability to other related tasks.

# Dataset
[Download the Datasets](http://www.patreo.dcc.ufmg.br/multi-view-datasets/)

# Citation:
If you find our work is useful, please kindly cite the following: BibTex

@article{zhao2023complementarity,
  title={When complementarity meets consistency: weighted collaboration fusion constrained by consistency between views for multi-view remote sensing scene classification},
  author={Zhao, Kun and Li, Senlin and Zhou, Lijian and Sun, Jie and Hao, Siyuan},
  journal={International Journal of Remote Sensing},
  volume={44},
  number={23},
  pages={7492--7514},
  year={2023},
  publisher={Taylor \& Francis}
}
