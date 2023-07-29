# Kernels-methods-2023-AMMI-Challenge
Team name: The Hunters

In this project, we propose many kernels methods from scratch for DNA Sequence classification.

The goal of the data challenge is to learn how to implement machine learning algorithms, gain understanding about them and adapt them to structural data.
For this reason, we have chosen a sequence classification task: predicting whether a DNA sequence region is binding site to a specific transcription factor.

Transcription factors (TFs) are regulatory proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes.
Genome-wide protein-DNA binding maps can be profiled using some experimental techniques and thus all genomics can be classified into two classes for a TF of interest: bound or unbound.
In this challenge, we will work with three datasets corresponding to three different TFs.

We design 5 kernels models:

- KernelSVM + Optuna + Crossvalidation
- Kernel Ridge Regression + Optuna + Crossvalidation
- Kernel Logistic Regression + Optuna + Crossvalidation
- Kernel Spectrum with SVM
- Kernel Mismatch with SVM 

We obtained a private score of 66.46% on the leaderboard and ranked 7th at the end of the competition.

The repository contains the solution for our best model(Kernel Mismatch with SVM)

This project was done in collaboration with [Binta Sow](https://github.com/BintaSOW1)

The [link](https://www.kaggle.com/competitions/kernel-methods-ammi-2023/overview) brings you to the hackathon website. 
