# ChemGCN
ChemGCN is a graph convolutional network to predict water solubilities of small molecules.

To get started, follow the steps below:

1. Install [Anaconda](https://www.anaconda.com/download).  

2. Install either the GPU or CPU ChemGCN environment.  
``
    conda env create --name chem_gcn --file environment_gpu.yml
``  
OR  
``  
    conda env create --name chem_gcn --file environment_cpu.yml
``  

3. Activate the environment.  
``
    conda activate chem_gcn
``  

4. Run the training script.  

``
python train_chemgcn.py
``
