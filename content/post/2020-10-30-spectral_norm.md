---
date: 2020-10-30
title: "Why I should 1 : Divide by spectral norm [ARTICLE IN PROGRESS DONT TAKE IT SERIOUSLY]"
tags : ["AI","Deep Learning","Analysis","Why I should"]

header:
  image: "spectre.jpg"

---
The idea of this series of very short articles (Why I should) is to explain some commonplace in machine learning.

Simply explain common tricks with mathematical arguments.

Let's dive into the **spectral norm** !

***

### The interest of the spectral norm

If you work on GAN or robustness, you probably deal with **spectral norm**.

* In GANs, the spectral norm is used on the discriminator and/or on the generator as in papers : [*Spectral Normalization for Generative Adversarial Networks*](https://arxiv.org/pdf/1802.05957.pdf) and [*Semantic Image Synthesis with Spatially-Adaptive Normalization*](https://arxiv.org/pdf/1903.07291.pdf). 

* In robustness, it is also used throughout the network to make it robust.

In both cases, the network weight matrices are **divided by their spectral norm**.The goal of this operation is to make **1-lipschitz** continuous the network.

### Lipschitz continuous application

If $(E,d_E)$ and $(F,d_F)$ are two metric spaces and $l: E \rightarrow F$  an application from $E$ to $F$,
$l$ is called **lipschtitz continuous** if : 
$$ \exists K > 0 \ |\  \forall x,y \in E, \ d_{F}\left(l\left(x\right), l\left(y\right)\right) \leq K d_{E}\left(x, y\right)$$

In practice $d_E$ and $d_F$ are norms and $l \in \mathcal{L}(E ; F)$ is a linear application. 

As $l$ is linear the Lipschitz condition can be rewritten as :

$$ \exists K > 0 \ | \ \forall x \in E, \ \Vert Wx \Vert_{\mathrm{F}} \leqslant K \Vert x \Vert_{\mathrm{E}}$$
with $W$ the associated matrix of the linear application $l$.

This equivalence is **only** true for **linear** applications.
It is natural to be interested only in the linear application. In fact it is only the weight matrices of the models that interest us here.

> How to interpret this property ? : we can see the norm of a vector as its energy $\mathcal{E}$. We can therefore rewrite Lipschitz's condition as :$$ \exists K > 0 \ | \ \forall x \in E, \  \frac{\mathcal{E}(Wx)}{\mathcal{E}(x)} \leqslant K  $$ In a sense **the energy ratio between input $x$ and output $Wx$ is bounded**, this property ensures that the energy $Wx$ **does not explode**. This makes our $W$ application more stable, more robust.

It is this property that is sought in GANs or to make its network robust.

Having lipschitz continuous applications ensures that our model is robust. The **lipschitz constant** $K$ allows us to manage the "**degree of robustness**" of our application.

We would like to have a stronger condition than the *$K$-lipschitz* continuity. In practice, most layers of neural networks are already $K$-lipschitzian applications for a certain $K$ . However, we would like to **make this constant K equal to 1** so that the output energy is smaller or equal to the input energy.

A solution to make our layer 1-lipschitz is to use the **spectral norm**.

### Spectral norm

As I said, the mathematical object that will make a network 1-Lipschitz continuous is the spectral norm.

Let $W \in M_{m,n}(\mathbb{R})$, the spectral norm of $W$ is defined as $$\sigma(W) = 
\sup _{\Vert x\Vert _{2} \leq 1}\Vert W x\Vert _{2} =
\sup _{x \neq 0} \frac{\Vert W x\Vert _{2}}{\Vert x\Vert _{2}}$$

To transform a linear application into a 1-lipschitz continuous application, simply divide the matrix W of the application by the spectral norm of this matrix : 

$$ W \leftarrow \frac{W}{\sigma(W)}$$
#### *Proof that  $\frac{W}{\sigma(W)}$ is Lipschitz continuous*
>$$
\begin{aligned}
&\forall x \neq 0,\ \frac{\left\Vert \frac{W}{\sigma(W)} x\right\Vert}{\Vert x \Vert}
= \frac{\Vert Wx \Vert}{\sigma(W)\Vert x \Vert}
= \frac{\left \Vert Wx\right \Vert}{ \Vert x \Vert} \frac{ \Vert z \Vert}{\left \Vert Wz\right \Vert}
= \frac{\frac{ \Vert Wx \Vert}{ \Vert x \Vert}}{\frac{ \Vert Wz \Vert}{ \Vert z \Vert}} \leq 1 \\\\
&with \ \frac{\left \Vert Wz\right \Vert }{ \Vert z \Vert }=\sup _{x \neq 0} \frac{ \Vert W x \Vert }{ \Vert x \Vert } \\\\
&\ and \ z= \underset{y \neq 0}{\arg \max } \frac{\Vert Wy \Vert}{\Vert y \Vert} \\\\
& \frac{\frac{ \Vert Wx \Vert}{ \Vert x \Vert}}{\frac{ \Vert Wz \Vert}{ \Vert z \Vert}} \leq 1 \Longleftrightarrow
 \frac{\Vert Wx \Vert}{\frac{ \Vert Wz \Vert}{ \Vert z \Vert}} \leq \Vert x \Vert \Longleftrightarrow
\Vert \frac{W}{\sigma(W)} x \Vert \leq \Vert x \Vert
\end{aligned}
$$
By definition it's mean $\frac{W}{\sigma(W)}$ is 1-Lipschitz continuous. ∎


### Apply spectral norm to a neural network
```python
import torch.nn as nn
m = nn.utils.spectral_norm(nn.Linear(20, 40))
```
Before going any further I will recall a few facts.

Let $ f_1: E_1 \rightarrow E_2, \ f_2: E_2 \rightarrow E_3, \ldots , f_N :E_N \rightarrow E_{N+1} $  be all Lipschitz applications. $$\forall i \in \\{1,2,\ldots,N\\}, \ \exists K_i > 0 \ | \ \forall x,y \in E_i, \ \Vert f_i(x)-f_i(y) \Vert_{E_{i+1}} \leq K_i \Vert x_i-y_i \Vert_{E_i}$$
$$ F=\bigotimes_{i=1}^{N} f_{i}=f_{1} \circ f_{2} \circ \cdots \circ f_{N}$$ therefore $F$ is $K$-lipschitz with $K = \prod_{n=1}^{N} K_i$

#### *Proof that  $F$ is $K$-lipschitz*
>$$
\begin{aligned} &\Big\Vert f_{N}\Big(f_{N-1} \cdots(f_{1}(x) \ldots\Big) - f_{N}\Big(f_{N-1} \cdots(f_{1}(y) \ldots\Big)  \Big\Vert \\\\
& \leq K_N \Big\Vert f_{N-1}\Big(f_{N-2} \cdots(f_{1}(x) \ldots\Big)  - f_{N-1}\Big(f_{N-2} \cdots(f_{1}(y) \ldots\Big)  \Big\Vert \\\\
& \qquad \qquad \qquad \qquad \qquad \qquad \qquad \vdots \\\\
& \leq \prod_{n=1}^{N} K_i \Vert x - y  \Vert
\end{aligned}
$$ 

By definition it's mean that $F$ is $\Big(\prod_{n=1}^{N} K_i \Big)$-lipschitz continuous. ∎