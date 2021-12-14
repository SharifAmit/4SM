# Installation 

## Pre-requisite
- Ubuntu 18.04 or later
- **Supports** : NVIDIA Pascal (P100, GTX10**), Volta (V100), Turing (GTX 16**, RTX 20**, Quadro)
- **Does not support** : NVIDIA Amphere (RTX 30**, A100) [In Development]

## 1. Download and Install Anaconda from the following link

```
https://www.anaconda.com/products/individual
```
Make sure to export your conda bin folder to .bashrc
```
export PATH="/root/anaconda3/bin:$PATH"
```

## 2. Open Anaconda Prompt from Start menu. Clone the repository from github. 
```
git clone https://github.com/SharifAmit/4SM.git
```

## 3. Enter the 4SM directory and type following in terminal to create a virtual environment with anaconda packages
```
cd 4SM
conda create -n streamlit-4sm anaconda python=3.7 -y
```
## 4. Activate the virual environment from the terminal.
```
conda activate streamlit-4sm
```
## 5. Run the bash script in the terminal.
```
sh install_linux.sh
```
## 6. Deactivate the virual environment
```
 source deactivate streamlit-4sm
```

# Running the app

## 1. Open Anaconda Prompt from Start menu 

## 2. Activate the virual environment
```
source activate streamlit-4sm
```
## 4. Type the following to run the app
```
streamlit src/run web_streamlit.py
```
## 5. A new browser will open with 4SM app running on it. 

## 6. Common installation issues

```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
Run
```
apt-get update && apt-get install -y python3-opencv
``