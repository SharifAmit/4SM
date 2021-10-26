## 1. Download and Install Anaconda from the following link

```
https://www.anaconda.com/products/individual
```

## 2. Open Anaconda Prompt from Start menu. 

## 3. Create a virtual environment with anaconda packages
```
create -n streamlit-calciumgan anaconda python=3.7
```
## 4. Activate the virual environment
```
conda activate streamlit-calciumgan
```
## 5. Install tensorflow-gpu 
```
conda install tensorflow-gpu=2.0.0
```
## 6. Install Keras
```
conda install keras=2.3.1
```
## 7. Install Streamlit & Streamlit-aggrid
```
pip install streamlit
pip install streamlit-aggrid
```
## 8. Install OpenCV
```
pip install opencv-python
```
## 9. Replace scikit-learn with a downgraded version
```
pip uninstall scikit-learn
pip install scikit-learn==0.20.4
```
## 10. install pycm
```
pip install pycm
```
