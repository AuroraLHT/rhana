# **Rh**eed **Ana**lysis

Author lists

An opensource library dedicated for analysis RHEED patterns and aim to extract information 
for phase mapping and crystal growth monitoring using machine learning algorithm. Cite this: 

## Installation
We haven't upload it to the Pypi yet, so mannul installation is required:
```bash
    git clone <project_url>
    cd <project>
    pip install -e .
```
The dependency should be all installed by pip automatically. Check the pytorch website on how to install a proper version 
 if you have issues related with fastai or pytorch.

## Philosophy

The idea of this package is to provide tools to quickly analysis the RHEED pattern while the experiment is done or even still running. The method described in the paper is demonstrated in several notebooks which illustrate how the information is extracted in step by step manner. The method works on RHEED video or even just RHEED of before and after deposition. The workflow is seperated into multiple components and is highly controlable. Users could use their observation/intuition to guide or even swap the output of one of the component. Despite the nature of RHEED is extremely complex, we wish to extract as much of structural information as possible for a totally unknown system. These information could be linked with other structural characterization technique to quickly realize the phase composition of novel materials and faciliate the connection between structure and properties.

## Notebooks

| Name | Description | What you would get |
| ---- | ----------- | ------------------ |
| ApplyUNetwithTracking | Use U-Net to generate feature masks for every rheed patterns and also center and crop out the region of interest | Masks in run-line encoding (RLE) that is exported to a csv file |
| DynamicPhaseChange | Observe how the pattern evolve over time | A  plot of variation in the relative intensity of each periodicity |
| Fastaiv2UNetTraining | Train an U-Net model to predict masks for spots and streaks | A UNet model that is  |
| MaterialMatching | Match materials from extracted periodicity | Phases that could be used to explain the RHEED pattern (heavily depend on experty in Materials Science) |
| PhaseMap-PhaseAnalysis | Extract the coexisting phases among a combi experiment with RHEED | All periodicity could be found from a combi experiemnt and the "phase" composition of each sample computed from the relative intensity of each periodicity |


## Training Data Preparation
Any general image labeling software would allow you to create mask for each image. Masks of each individual features could be stored also depending on the software implementation. The software used in this paper is [django-labeller](https://github.com/Britefury/django-labeller) which is a user friendly open-source software.