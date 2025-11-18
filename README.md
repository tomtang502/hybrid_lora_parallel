# On the Optimization of Parallel Training of Large Model with Hybrid Attention and Sparse Activated Parameters

---

Authors: Tom Tang, Tony Tang

[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/) [![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org) [![C++](https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://isocpp.org/) [![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://www.nvidia.com/)

<!-- [![Transformers](https://img.shields.io/badge/Transformers-🤗-yellow?style=for-the-badge)](https://huggingface.co/docs/transformers) -->

<!-- [![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE) -->

<!-- [![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)](https://ubuntu.com/) -->

<hr style="border: none; height: 1px; background-color: #17f1baff; margin: 0;">

## Updates

- 25/11/17: Proposal Uploaded
- 25/11/16: Initialize the repo
<hr style="border: none; height: 1px; background-color: #17f1baff; margin: 0;">

## Training Pipeline Roadmap

![#0069e0](https://singlecolorimage.com/get/0069e0/12x12) Classic PyTorch Pipeline &nbsp;&nbsp;
![#ed5103](https://singlecolorimage.com/get/ed5103/12x12) Targeted Feature &nbsp;&nbsp;
![#999999](https://singlecolorimage.com/get/999999/12x12) Advanced Feature

<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td align="left">
      <img src="https://singlecolorimage.com/get/0069e0/12x12" width="12" height="12" />
      &nbsp;<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html">Distributed Data Parallel</a>
    </td>
    <td>─┐</td>
  </tr>
  
  <tr>
    <td align="left">
      <img src="https://singlecolorimage.com/get/0069e0/12x12" width="12" height="12" />
      &nbsp; <a href="https://docs.pytorch.org/docs/stable/fsdp.html">ZeRO-3 Sharding</a>
    </td>
    <td>──►</td>
    <td>
      <img src="https://singlecolorimage.com/get/ed5103/12x12" width="12" height="12" />
      &nbsp;Distributed Tensor Sharding with Data Parallel (FSDP2)
    </td>
    <td>──►</td>
    <td>
      <img src="https://singlecolorimage.com/get/ed5103/12x12" width="12" height="12" />
      &nbsp;Pipeline Parallelism (Uniform Parameters Split)
    </td>
    <td>─┐</td>
  </tr>

  <tr>
    <td></td><td></td><td></td><td></td><td></td>
    <td align="right">⬇︎</td>
  </tr>

  <tr>
    <td>
      <img src="https://singlecolorimage.com/get/999999/12x12" width="12" height="12" />
      &nbsp;LoRA CUDA Kernel
    </td>
    <td>◄──</td>
    <td>
      <img src="https://singlecolorimage.com/get/999999/12x12" width="12" height="12" />
      &nbsp;Tensor Parallelism
    </td>
    <td>◄──</td>
    <td>
      <img src="https://singlecolorimage.com/get/ed5103/12x12" width="12" height="12" />
      &nbsp;Pipeline Parallelism (Dynamic Parameters Split)
    </td>
    <td align="right">─┘</td>
  </tr>
</table>
