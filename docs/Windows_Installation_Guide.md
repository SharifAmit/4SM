# Installation 

## Pre-requisite
- Windows 7 or later
- **Supports** : NVIDIA Pascal (P100, GTX10**), Volta (V100), Turing (GTX 16**, RTX 20**, Quadro)
- **Does not support** : NVIDIA Amphere (RTX 30**, A100) [In Development]

## 1. Download and Install Anaconda from the following link
```
https://www.anaconda.com/products/individual
```

## 2. Open Anaconda Prompt from Start menu. Clone the repository from github. 
```
git clone https://github.com/SharifAmit/4SM.git
```

## 3. Enter the 4SM directory and type following in terminal to create a virtual environment with anaconda packages
```
cd 4SM
conda create -n streamlit-4sm m2-bash anaconda python=3.7 -y
```
## 4. Activate the virtual environment from the terminal.
```
conda activate streamlit-4sm
```
## 5. Run the bash script to install required python libraries
```
bash install_windows_libs.sh
```
alternatively you can copy and paste the following commands in your terminal
```
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
```

# Running the application
To run you the application, make sure to have the anaconda envire activated
 in the main projcet

## 1. Open Anaconda Prompt from Start menu 

## 2. Activate the virual environment
```
conda activate streamlit-4sm
```
## 4. Type the following to run the app
```
streamlit run src/web_streamlit.py
```
## 5. A new browser will open with 4SM app running on it. 

