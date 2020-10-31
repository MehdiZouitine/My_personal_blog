---
date: 2020-10-30
title: "Why I should 1 : Divide by spectral norm"
tags : ["AI","Deep Learning","Analysis","Why I should"]

header:
  image: "spectre.jpg"

---
The idea of this series of very short articles (Why I should) is to explain some commonplace in machine learning.

Simply explain common tricks with mathematical arguments.

Let's dive into the **spectral norm** !

***

### The interest of the spectral norm

The spectral norm is used in several areas such as image generation with **GANs** or in the field of **Robustness**.

* In GANs, the spectral norm is used on the discriminator and/or on the generator as in papers : [*Spectral Normalization for Generative Adversarial Networks*](https://arxiv.org/pdf/1802.05957.pdf) and [*Semantic Image Synthesis with Spatially-Adaptive Normalization*](https://arxiv.org/pdf/1903.07291.pdf). 

* In robustness, it is also used throughout the network to make it robust.

In both cases, the network weight matrix is **divided by its spectral norm**.The goal of this operation is to make lipschitz continuous the network. And more precisely to make lipschitz continuous the linear application associated with the weight matrix of the network.

### Lipschitz continuous application

In this sub-section we will only talk about **linear** lipschitzian applications.
We are only interested in this class of function because it is the spectral norm of the weight **matrix** that interests us.

If $(E,d_E)$ and $(F,d_F)$ are two metric spaces and $l: E \rightarrow F$  an application from $E$ to $F$,
$l$ is called **lipschtitz continuous** if : 
$$ \exists K > 0 \ |\  \forall x,y \in E, \ d_{F}\left(l\left(x\right), l\left(y\right)\right) \leq K d_{E}\left(x, y\right)$$

In practice $d_E$ and $d_F$ are norms and $l \in \mathcal{L}(E ; F)$ is a linear application. 

As $l$ is linear the Lipschitz condition can be rewritten as :

$$ \exists K > 0 \ | \ \forall x \in E, \ \Vert Wx \Vert_{\mathrm{F}} \leqslant K \Vert x \Vert_{\mathrm{E}}$$
with $W$ the associated matrix of the linear application $l$.

This equivalence is **only** true for **linear** applications.
It is natural to be interested only in the linear application. In fact it is only the weight matrices of the models that interest us here.

> *How to interpret this property :We can see the norm of a vector as its energy $\mathcal{E}$.We can therefore rewrite Lipschitz's condition as :$$ \exists K > 0 \ | \ \forall x \in E, \  \mathcal{E}(Wx) \leqslant K \mathcal{E}(x) $$*






 \propto