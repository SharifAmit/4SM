<h4 align="center">
  4SM: Subcellular Signal Segmenting Spatiotemporal Model
</h4>

<div align="center">
  <a href="#installation-guide"><b>Installation Guide</b></a> |
  <a href="#usage"><b>Usage Guide</b></a> |
  <a href="https://github.com/SharifAmit/4SM/tree/main/examples/image_dataset/"><b>Test Dataset</b></a> |
  <a href="https://www.youtube.com/watch?v=t2LsQkyAGQc" target="4SM tutorial"><b>4SM Installation and Demo on YouTube</b></a>
</div>

<br/>

# 4SM: Subcellular Signal Segmenting Spatiotemporal Model

This code is part of our paper **4SM: New open-source software for subcellular segmentation and analysis of spatiotemporal fluorescence signals using deep learning** and is currently under review. 

The authors of the papers are <b>Sharif Amit Kamran, Khondker Fariha Hossain, Hussein Moghnieh, Sarah Riar, Allison Bartlett, Alireza Tavakkoli, Kenton M Sanders and Salah A. Baker</b>

The code is authored and maintained by Sharif Amit Kamran [[Webpage]](https://www.sharifamit.com/) and Hussein Moghnieh [[Webpage]](https://medium.com/@husseinmoghnie).


# Abstract

To understand cellular dynamics in fluorescence imaging, we need a fast, accurate, and reliable software or tool. In recent times, Deep learning has advanced biomedical image analysis and consistently achieved state-of-the-art accuracy by learning from high volumes of data. Despite these advances, there has been little to no application in the segmentation of subcellular fluorescein signals. Spatio-temporal maps (STMaps) are a transformed version of dynamic cellular signals recordings, visualized by an image of a function of time and space. Current approaches of segmentation and quantification of these images are time-consuming and require an expert annotator. To alleviate this, we propose an open-source software called "4SM" that incorporates a novel deep-learning methodology to segment subcellular fluorescein signals with high accuracy. Moreover, the tool provides a fast, robust, and consistent data analysis and retrieval pipeline that can accommodate different patterns of signals across multiple cell types. In addition, the software allows the user to experience seamless data accessibility, quantification, graphical visualization and allows high throughput for large datasets. 

[![IMAGE ALT TEXT HERE](docs/graphical_abstract.png)](https://www.youtube.com/watch?v=t2LsQkyAGQc)


# Installation Guide

## Pre-requisite

- CUDA version 10+
- List of NVIDIA Graphics cards supporting CUDA 10+
      https://gist.github.com/standaloneSA/99788f30466516dbcc00338b36ad5acf

## Installing and Running 4SM
- [Windows](docs/Windows_Installation_Guide.md)
- [Linux 64 / Ubuntu 18](docs/Ubuntu_Instllation_Guide.md)  
- [Google Colab](https://colab.research.google.com/drive/1mlmrOho8D5Cd-eqlV-aZHAYAY-EpEjmj?usp=sharing)

## Installation Instructions and Demo on YouTube
[![IMAGE ALT TEXT HERE](docs/youtube_graphical_abstract.png)](https://www.youtube.com/watch?v=t2LsQkyAGQc)


# Usage
4SM pris web-based easy to use graphical interface for subcellular segmentation and analysis of spatiotemporal fluorescence signals using deep learning

###Input:
 - One or multiple images to process
   - **NOTE: 4SM supports only 8-bit grey scale images**
 - Set the threshold, crop size, and calibration parameters

###Output:
 - Segmented images
 ![](docs/Image_Segmentation.png)  

 - Quantification results of the segmented images
  ![](docs/4SM_stochastic.png)  

### Control Panel
![](docs/control_panel.png)  

### Image Segmentation
![](docs/Image_Segmentation.png)  

### Quantification



# License
The code is released under the GPL-2 License, you can read the license file included in the repository for details.
