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
git clone https://github.com/SharifAmit/CalciumGAN.git
```

## 3. Enter the CalciumGAN directory and type following in terminal to create a virtual environment with anaconda packages
```
cd CalciumGAN
conda create -n streamlit-calciumgan m2-bash anaconda python=3.7 -y
```
## 4. Activate the virual environment from the terminal.
```
conda activate streamlit-calciumgan
```
## 5. Run the bash script in the terminal.
```
bash windows.sh
```
## 5. Deactivate the virual environment
```
conda deactivate streamlit-calciumgan
```

# Running the app

## 1. Open Anaconda Prompt from Start menu 

## 2. Activate the virual environment
```
conda activate streamlit-calciumgan
```
## 4. Type the following to run the app
```
streamlit run CalciumGAN/web_streamlit.py
```
## 5. A new browser will open with CalciumGAN app running on it. 

