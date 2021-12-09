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
cd CalciumGAN
conda create -n streamlit-4sm m2-bash anaconda python=3.7 -y
```
## 4. Activate the virual environment from the terminal.
```
conda activate streamlit-4sm
```
## 5. Run the bash script in the terminal.
```
bash windows.sh
```
## 6. Deactivate the virual environment
```
conda deactivate streamlit-4sm
```

# Running the app

## 1. Open Anaconda Prompt from Start menu 

## 2. Activate the virual environment
```
conda activate streamlit-4sm
```
## 4. Type the following to run the app
```
streamlit run 4SM/web_streamlit.py
```
## 5. A new browser will open with 4SM app running on it. 

