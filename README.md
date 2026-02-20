# ultrasound_vl_csa_and_ei
Automated Analysis of Vastus Lateralis Cross Sectional Area and Echo Intensity from Panoramic Ultrasound

Models and MATLAB GUIs: https://huggingface.co/kenziew/ultrasound_vl_csa_and_ei  

Manuscript:  
McKenzie S. White, Arimitsu Horikawa-Strakovsky, Kirby P. Mayer, Brian W. Noehren, Yuan Wen,
Open-Source AI for Vastus Lateralis and Adipose Tissue Segmentation to Assess Muscle Size and Quality,
Ultrasound in Medicine & Biology  

DOI: https://doi.org/10.1016/j.ultrasmedbio.2025.08.008

Please see MATLAB_VL_CSA_documentation_v2.pdf for detailed information on usage.   

Here we release a validated model for detecting the subcutaneous adipose tissue
and vastus lateralis muscle in panoramic B-mode Ultrasound Images. We believe that
alongside models, there is also a need to release open-sourced tools that apply that model
to unseen images and permit adjustments of ROI’s for end-users. In addition, those tools
should be user-friendly, well-documented, and have automated options for processing and
compiling large datasets. While we are currently working on building models with larger
datasets and adding more features to our GUIs, here we release our model, images, and
GUI’s built in MATLAB and python (NAPARI) for researchers to implement assessments of ultrasound with ease.
Our hope is that researchers will utilize these tools and contribute images to build larger
models. We are happy to help you get up and running if you run into issues. Feel free to
reach out!

McKenzie S. White  
Email: kenzieswhite@gmail.com  

Arimitsu Horikawa-Strakovsky  
Email: arimitsu0803@gmail.com  
