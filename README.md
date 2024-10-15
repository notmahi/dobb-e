![preview](https://github.com/notmahi/dobb-e/assets/3000253/341faa2f-285a-4152-91f6-73bec2811a97)
# Dobb·E

[![arXiv](https://img.shields.io/badge/arXiv-2311.16098-163144.svg?style=for-the-badge)](https://arxiv.org/abs/2311.16098)
![License](https://img.shields.io/github/license/notmahi/bet?color=873a7e&style=for-the-badge)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-262626?style=for-the-badge)](https://github.com/psf/black)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-db6a4b.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)

[Project webpage](https://dobb-e.com) · [Documentation (gitbooks)](https://docs.dobb-e.com) · [Paper](https://arxiv.org/abs/2311.16098)

**Authors**: [Mahi Shafiullah*](https://mahis.life), [Anant Rai*](https://raianant.github.io/), [Haritheja Etukuru](https://haritheja.com/), [Yiqian Liu](https://www.linkedin.com/in/eva-liu-ba90a5209/), [Ishan Misra](https://imisra.github.io/), [Soumith Chintala](https://soumith.ch), [Lerrel Pinto](https://lerrelpinto.com)

Open-source repository of the hardware and software components of [Dobb·E](https://dobb-e.com) and the associated paper, [On Bringing Robots Home](https://arxiv.org/abs/2311.16098)

https://github.com/notmahi/dobb-e/assets/3000253/d332fa42-d351-4ad0-a305-4b6664bd7170

<details>
  <summary><h2>Abstract</h2></summary>
  Throughout history, we have successfully integrated various machines into our homes - dishwashers, laundry machines, stand mixers, and robot vacuums are a few of the latest examples. However, these machines excel at performing a single task effectively. The concept of a “generalist machine” in homes - a domestic assistant that can adapt and learn from our needs, all while remaining cost-effective has long been a northstar in robotics that has been steadily pursued for decades. In this work, we initiate a large-scale effort towards this goal by introducing Dobb·E, an affordable yet versatile general-purpose system for learning robotic manipulation within household settings. Dobb·E can learn a new task with only five minutes of a user showing it how to, thanks to a demonstration collection tool (“The Stick”) we built out of cheap parts and iPhones. We use the Stick to collect 13 hours of data in 22 homes of New York City, and train Home Pretrained Representations (HPR). Then, in a novel home environment, with five minutes of demonstrations and fifteen minutes of adapting the HPR model, we show that Dobb·E can reliably solve the task on the Stretch, a mobile robot readily available in the market. Across roughly 30 days of experimentation in homes of New York City and surrounding areas, we test our system in 10 homes, with a total of 109 tasks in different environments, and finally achieve a success rate of 81%. Beyond success percentages, our experiments reveala plethora of unique challenges absent or ignored in lab-robotics, ranging fromeffects of strong shadows, to demonstration quality by non-expert users. With the hope of accelerating research on home robots, and eventually seeing robot butlers in every home, we open-source Dobb·E software stack and models, our data, and our hardware designs.
</details>

## What's on this repo
Dobb·E is made out of four major components:
1. A hardware tool, called [The Stick](https://dobb-e.com/#hardware), to comfortably collect robotic demonstrations in homes.
2. A dataset, called [Homes of New York (HoNY)](https://dobb-e.com/#dataset), with 1.5 million RGB-D frames. collected with the Stick across 22 homes and 216 environments of New York City.
3. A pretrained lightweight foundational vision model called [Home Pretrained Representations (HPR)](https://dobb-e.com/#models), trained on the HoNY dataset.
4. Finally, the platform to tie it all together to [deploy it in novel homes](https://dobb-e.com/#videos), where with only five minutes of training data and 15 minutes of fine-tuning HPR, Dobb·E can solve many simple household tasks.

Reflecting this structure, there are four folders in this repo, where:
1. [`hardware`](hardware) contains our 3D printable STL files, as well as instructions on how to set up the Stick.
2. [`stick-data-collection`](stick-data-collection) contains all the necessary software for processing any data you collect on the Stick.
3. [`imitation-in-homes`](imitation-in-homes) contains our code for training a policy on your collected data using our pretrained models, and also the code to pretrain a new model yourself.
4. [`robot-server`](robot-server) contains the code to be run on the robot to deploy the learned policies.

The primary documentation source is gitbooks at [https://docs.dobb-e.com](https://docs.dobb-e.com). There are also associated documentations inside each folder's READMEs.

## Paper
![paper_preview](https://github.com/notmahi/dobb-e/assets/3000253/0190c36b-da84-4b77-9979-762062c3b2b7)
Get it from [ArXiv](https://arxiv.org/abs/2311.16098) or our [website](https://dobb-e.com/#paper).


## Citation
If you find any of our work useful, please cite us!
<pre>
@article{shafiullah2023bringing,
  title={On bringing robots home},
  author={Shafiullah, Nur Muhammad Mahi and Rai, Anant and Etukuru, Haritheja and Liu, Yiqian and Misra, Ishan and Chintala, Soumith and Pinto, Lerrel},
  journal={arXiv preprint arXiv:2311.16098},
  year={2023}
}
</pre>
