call conda create --prefix ./4conda python=3.7
call conda install -p ./4conda -y tensorflow-gpu=2.0.0
call conda activate ./4conda
call pip install tensorflow-estimator==2.0.0
call conda install -p ./4conda -y keras=2.3.1
call pip install streamlit==0.86.0
call pip install streamlit-aggrid==0.2.1
call pip install opencv-python
call pip uninstall -y scikit-learn
call pip install scikit-learn==0.20.4
call pip install pycm
call pip install h5py==2.10.0 --force-reinstall
call pip install matplotlib==3.5.1
call pip install seaborn==0.11.2

PAUSE