---
date: 2020-06-21
title: Multi-agents Reinforcement Learning
tags : ["IA","RL","Markov"]
header:
  image: "robots.jpg"
  
---
# Understand Multi Agent Reinforcement learning (MARL) : An introduction to Grid Wise Control

***

Hi ! In this post I will explain to you one of the most efficient method for multi-agent reinforcement learning. This method is derived from paper :
[*Grid-Wise Control for Multi-Agent Reinforcement Learning in Video Game AI*](http://proceedings.mlr.press/v97/han19a/han19a.pdf).

Before reading this I advise the reader to have some notion of reinforcement learning (policy,reward,state...), deep learning, markov process, and optimisation.



Let's go into a little more detail! :smiley:
***
## A quick overview of ways of looking at the MARL problem



When you're looking to train multiple agents to solve a task, you need to ask yourself how you're going to pose the problem.

- Will I consider all my agents as a single agent, i.e. have a global policy, status and rewards for all my agents  
> It's what we  called **centralized learning**, centralized learning is a kind "change of variables" where we consider that our agent is the union of all our agents. This way of looking at the problem allows us to **maximize the coordination** of our agents. But do you see the problem in this way? Imagine that we have $N$ agents with each $K$ action then the cardinality of the action space will be $K^N$ action. In sum the join-action space is to large and a slight increase in the number of agents would increase the size of this space exponentially.





- Or will I consider all my agents independently ?
> It's what we  called **decentralized learning** ,in decentralized learning each agents learn his own policy based on his "local observation-action trajectory" (see [Independant Q-learning](http://web.media.mit.edu/~cynthiab/Readings/tan-MAS-reinfLearn.pdf)).As you can imagine, this type of learning has difficulty modeling communication between agents.


Of course there are methods between these two visions which are a mixture of centralized and decentralized learning. *Nothing's all black and white, it's the gray that wins.*
This article does not aim to explain all the learning methods (you can find references to methods (centralized, decentralized and mix in the original paper).

- The important thing to remember is that these solutions have one major flaw :
>  "For many multi-agent settings, the **number of agents acting
in the environment keeps changing both within and across
episodes**. For example, in video games, the agents may die
or be out of control while new agents may join in, e.g., the
battle game in StraCraft. Similarly, in real-world traffic,
vehicles enter and exit the traffic network over time, inducing complex dynamics. Therefore, a main challenge is to
**flexibly control an arbitrary number of agents and achieve
effective collaboration at the same time**. Unfortunately, all
the aforementioned MARL methods suffer from **trading-off
between centralized and decentralized learning to leverage
agent communication and individual flexibility**. Actually,
most of the existing MARL algorithms make a default assumption that the number of agents is **fixed before learning**.
Many of them adopt a well designed reinforcement learning
structure, which, however, depends on the number of agents."

***
## A solution: The Grid-Wise Control

Let's get to the heart of the matter!

We will take as an example of Starcraft2 game :

 Starcraft 2 is a strategy game where you have to control and cooperate several agents (often hundreds) to destroy the enemy base : *How do we use these agents and get them to cooperate in destroying the enemy base? ?*


To bring a solution to the MARL problem we define a well known architecture in deep learning: an **encoder-decoder :hourglass_flowing_sand:** .

We will present this architecture layer by layer in a rather static way. Then we will see how it works and the different algorithms it uses to train the agents.



{{< figure library="true" src="GW_archi.png" title="grid-wise control architecture" lightbox="true" >}}


### Input layer :


* **Input tensor** : The state grid $s \in \mathbb{R}^{w \times h \times c_{s}}$ where $w$ is the width of the grid, $h$ is the height and $c_{s}$ the features number. We can see this tensor as a stack of $c_s$ feature maps on our agents and our environment.

Let's take an example: Let's imagine that we want to train a set of agents to destroy the ennemy bases.

{{< figure library="true" src="scbase.jpg" title="Starcraft game situation" lightbox="true" >}}

Our architecture can't take that image. We have to find a way to encode the relevant information of the environment and the agents in the form of a set of grids (a tensor).

To describe our image we can give a set of grids as the following image:

{{< figure library="true" src="scbase2.png" title="Starcraft game situation" lightbox="true" >}}


Each feature map (to the right of the image) represents relevant information about the environment and the local environment of our agents. Some information present on the initial image is deliberately hidden because it is useless.

For example the feature map on the first line in the 4th position represents the player's camera. This is a more than relevant information!

This way we can create full feature tensors that represent the current state of our agents in the environment. We can then give this tensor to our network.

### Encoder block :

* **Convolutional encoder** : The state $s$ (our features tensor, a stack of feature maps) is fed to a convolutional network. This will create a latent representation of our state s (our feature tensor). This representation perfectly summarizes our state .
**"It also naturally handles agent collaboration, because the stacked convolutional layers can provide
sufficiently large receptive field for the agents to communicate. The GridNet also enables fast parallel exploration,
because experiences from one agent are immediately transferred to others through the shared convolutional parameters."**
The latent vector resulting from these successive convolutions (called encoding on the image) is a perfect summary of the state $s$, indeed it takes into account the situation of each agent as well as the collective dynamics.

**convolutional encoder image**


### Decoder block :

* **Convolutional Decoder**: The embedding $s_{encoded}$ of our state $s$ is fed into a Deconvolutional network. This will decod the latent representation of our state $s$. I remind you that this state concentrates the information of each agent as well as their collective dynamics. We want to decode this latent vector and give it a form of action grid (we will see this in the next paragraph). To do this we proceed to deconvolution and upsampling operations to increase the dimension of this vector. These upsampling operations result in an action grid $a \in \mathbb{R}^{w \times h \times c_{a}}$ where $c_{a}$ is the actions number.
For example if our agents are in a square zone of 20 length and that each agent has 4 actions then the action grid will have the following dimension $(20 \times 20 \times 4)$.

### Critic block

As you can see, our architecture has another element. Our network is divided in two: the encoder-decoder and what we will call the **critic block**.Once the information is encoded, we will extract via a fully connected layer the **value function**. We will explain what this function is in the next part which will deal with the optimization of this network.


From now on, our architecture is well defined ! The question that comes from now on is: *How to optimize our network to answer the MARL problem?* With the **policy gradient**!
***
## Network optimisation

### A few obvious facts

Let's summarize what we just said with this little diagram!


{{< figure library="true" src="grid_wise_diag.svg" title="Grid-wise control diagram" lightbox="true" >}}


As you have just seen, this diagram adds extra information! What I called 'critical-output-value' and also the 'policy optimisation'.

Our architecture has **2 outputs**: The action grid **$a$** and the output value **$v$**. To optimize our network we must therefore minimize a loss function. Naturally this function will take in argument the 2 outputs of our architecture and the current state $s$ :
$$loss(s,a,v) = \psi(s,a,v)$$



The optimization of this function corresponds to the last step of our diagram !

The question that comes naturally now is the **choice of the $\psi$ function.**

### Policy optimisation and Actor-critic

The purpose of our architecture is to give us for a state the best possible action (to maximize the reward). Our architecture therefore serves to optimize what we call **policy** $\pi$.

>The policy is a function that gives us the probability of making an action **$a$** knowing that we are in a certain state **$s$**. We want to optimize this policy so that our agent does the best sequence of actions (i.e. the sequence of actions that maximizes the reward).

Our loss function $\psi$ will therefore be a function that optimizes this policy.

{{< figure library="true" src="actor_critic.svg" title="optimization by actor-critic policy gradient" lightbox="true" >}}
