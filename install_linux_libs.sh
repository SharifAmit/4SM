#!/bin/bash
conda install -y tensorflow-gpu=2.0.0
pip install tensorflow-estimator==2.0.0
conda install -y keras=2.3.1
pip install streamlit==0.86.0
pip install streamlit-aggrid==0.2.1
pip install opencv-python
pip uninstall -y scikit-learn
pip install scikit-learn==0.20.4
pip install pycm
pip install h5py==2.10.0 --force-reinstall
pip install matplotlib==3.5.1
pip install seaborn==0.11.2