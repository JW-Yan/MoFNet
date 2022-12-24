# MoFNet: Deep multi-omic network fusion for marker discovery of Alzheimer’s Disease

MoFNet v1.0
Yan's Lab, Indiana University Purdue University Indianapolis
Developed by Yash Raj and Linhui Xie

## Description
MoFNet is an interpretable multi-omic graph neural network approach for integratively identifying network-level biomarkers of target diseases. This repository was initially forked from Varmole(https://github.com/namtk/Varmole). Instead of having single transparent layer to integrate quantitative trait loci (QTLs) and gene regulatory networks (GRNs) into prior biological knowledge, this MoFNet has further extended the network structure to two transparent layers to integrate the multi-omic(eg. genomics, transcriptomics and preteomics) dataset into deep neural network. More transparent layers are expected if more layers of multi-omic features are available. The preprint is posted on bioarxiv: https://www.biorxiv.org/content/10.1101/2022.05.02.490336v1.


## Installation
The following libraries are required,
Python >= 3.6
pandas >= 1.2
numpy >= 1.20 
PyTorch >= 0.5
captum >= 0.3.0


## Usage
To have the multi-omic matrix data ready in a .csv format. Each row contain the multi-omic data for single participant. The column number is equal to the entire number of multi-omic features. For adjacency matrix, the row and column numbers are identical to the entire number of multi-omic features, and it should be in a .csv format as well. All input files are supposed to put under the same folder. Then, ran the code in terminal, in the format shown below,
$ python mofnet_main.py /the_path_to_your_dataset/

There is anther short pipeline provided in mofnet_example.ipynb to run MoFNet interactively.

## Contents
You can request access the real dataset through AD knowledge portal(https://adknowledgeportal.synapse.org) with synapse #.
For the detailed information under this repository, the root folder contains following scripts,
mofnet_main.py contains the main structure.
model.py contains all input information to the pipeline .
train.py contains the training part of this method.
utils.py contains all utility functions that will be applied in the approach.


## Information
 * LR: learning rate to this structure.
 * L1REG: L1 norm regularization penalty.


## License
MIT License

Copyright (c) 2022

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
