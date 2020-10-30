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

The spectral standard is used in several areas such as image generation with **GANs** or in the field of **Robustness**.

* In GANs, the spectral norm is used on the discriminator and/or on the generator as in papers : [*Spectral Normalization for Generative Adversarial Networks*](https://arxiv.org/pdf/1802.05957.pdf) and [*Semantic Image Synthesis with Spatially-Adaptive Normalization*](https://arxiv.org/pdf/1903.07291.pdf). 

* In robustness, it is also used throughout the network to make it robust.

In both cases, the network weight matrix is **divided by its spectral norm**.

The goal of this operation is to make the lipschitz network. And more precisely to make lipschitz the linear application associated with the weight matrix of the network.