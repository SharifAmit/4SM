# Installation 

## Pre-requisite
- Windows 7 or later
- **Supports** : NVIDIA Pascal (P100, GTX10**), Volta (V100), Turing (GTX 16**, RTX 20**, Quadro)
- **Does not support** : NVIDIA Amphere (RTX 30**, A100) [In Development]

### 1. Download and Install Anaconda from the following link
```
https://www.anaconda.com/products/individual
```
Important: Make sure to select "Add Anaconda(3) to the system Path environment variable" during installation as shown below


![](anaconda_path.png)  

### 2. Download and unzip  the application from

[Click here to download 4SM application](https://github.com/SharifAmit/4SM/archive/refs/heads/st5.zip)


### 3. Install Python libraries dependencies 

In the root folder of the unzipped project, double click on the following script
```
install_windows_libs.bat
```
This step will create Anaconda's environment and install all python dependencies


# Running the application
To run your application, simply double click on the batch script in the root folder of the project
```
start_app_windwos.bat
```
A new browser will open with 4SM app running on it on
[http://localhost:8501](http://localhost:8501)

Please note that if your browser is an old internet explorer, copy the above web address and paste it in a Google Chrome or Microsoft Edge

# Troubleshooting
 
#### 1-Anaconda path cannot be found
If double clicking on the batch files is not working because it cannot find Anaconda executable path, you can always run Anaconda bash mode as shown below

- Activate "Anaconda prompt"

![](anaconda_prompt.png)  

- Navigate to the root folder of the application and install the depdencies by running ```install_windwos_libs.bat```

- Run the application by running ```start_app_windows.bat```

#### 2-Blank page when opening the application

Make sure you are not using an older version of internet explorer. It is recommended to use Google Chrome, Microsoft Edge, or FireFox
