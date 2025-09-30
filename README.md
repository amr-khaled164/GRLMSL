# GRLMSL: Graph Reinforcement Learning-Based Delay-Aware Microservice Instance Selection and Load Balancing in Edge Computing

## Model Overview

**GRLMSL** is a novel model designed to tackle the critical challenge of selecting optimal microservice instances for IoT requests within dynamic edge computing environments. These environments are characterized by:

- **Time-varying network conditions**
- **Numerous interdependent microservice instances**
- **Dynamic request generation patterns**

The model intelligently navigates this complexity to minimize request dealy and achieve efficient load balancing.

## Key Contributions

The primary contributions of this work are as follows:

### 1. Edge-Enhanced Graph Attention Network
- **Description:** Enhances the standard Graph Attention Network (GAT) by incorporating specific edge features, allowing the model to better capture the complex relationships and states between interdependent microservice instances and edge servers.
- **Relevant Codes:** EGAT.py, GAT.py

### 2. Action Space Partition and Invalid Action Masking
- **Description:** Effectively manages the large and complex action space by partitioning it into logical segments. This is combined with invalid action masking to prevent the selection of unavailable or inappropriate microservice instances, significantly improving learning efficiency and stability.
- **Relevant Codes:** AdvancedAgent.py

### 3. Improved Action Exploration with Noisy Networks
- **Description:** Integrates Noisy Networks into the reinforcement learning agent to replace the conventional epsilon-greedy strategy. This promotes more sophisticated and effective exploration of the action space, leading to better policy discovery.
- **Relevant Codes:** NoisyNet.py

### 4. Enhanced Experience Replay with Gradient Surgery
- **Description:** Improves the experience replay mechanism by integrating gradient surgery techniques. This approach helps to mitigate the issue of conflicting gradients when learning from a batch of diverse experiences, resulting in more stable and convergent training.
- **Relevant Codes:** PCGrad.py, Memory.py, SumTree.py

## Environment
We develop our codes in the following environment using PyCharm
- **CUDA==12.6**
- **Python==3.12.7**
- **torch==2.7.0+cu126**

## How to run the codes
Please run the codes in the SystemPlatform.py file.
```python
if __name__ == "__main__":

    args = get_options() 
    sp = SP(args)
    sp.train_agent() 
    sp.train_agent_more(20, 4) 
    sp.test_agent(23) 
```

