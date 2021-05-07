# Anonymize-DDP

Pseudonimizing software for data download packages (DDP), specifically focussed on Instagram.

## Table of Contents
* [About Anonymize-DDP](#about-anonymize-ddp)
  * [Built with](#built-with)
  * [License](#license)
  * [Attribution and academic use](#attribution-and-academic-use)
* [Getting Started](#getting-started)
  * [Preparatory steps](#preparatory-steps)
    * [Clone repository](#clone-repository)
    * [Download DDP](#download-ddp)
    * [Create additional files](#create-additional-files)
  * [Run software](#run-software)
  * [Evaluation](#evaluation)
  
## About Anonymize-DDP
**Date**: December 2020

**Researchers**:
* Laura Boeschoten (l.boeschoten@uu.nl)

**Research Software Engineers**:
* Martine de Vos (m.g.devos@uu.nl)
* Roos Voorvaart (r.voorvaart@uu.nl)

### Built With

The blurring of text in images and videos is based on a pre-trained version of the [EAST model](https://github.com/argman/EAST). Replacing the extracted sensitive info with the pseudonimized substitutes in the data download package is done using the [AnonymoUUs](https://github.com/UtrechtUniversity/anonymouus) package.

### License

The code in this project is licensed with [MIT](LICENSE.md).

### Attribution and academic use
The manuscript detailing the first release of anonymize-ddp is submitted and available [here](https://arxiv.org/pdf/2105.02175.pdf)

A data set consisting of 11 personal Instagram archives, or Data-Download Packages, was created to [evaluate](/src/evaluation) the anonymization procedure.
This data set is publicly available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4472606.svg)](https://doi.org/10.5281/zenodo.4472606)


# Getting Started

## Preparatory steps

Before running the software, the following steps need to be taken:

1. **[Clone repository](#clone-repository)**
2. **[Download DDP](#download-ddp)**
3. **[Create additional files](#create-additional-files)**

### Clone repository

To clone this repository, you'll need *Git installed* on your computer. When Git is installed, run the following code in the command line:

```
# Clone this repository
$ git clone https://github.com/UtrechtUniversity/anonymize-ddp

# Go into the repository
$ cd anonymize-ddp

# Install dependencies
pip install -r requirements.txt

# When experiencing difficulties
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
N.B. When experiencing difficulties with installing torch, have a look at the [PyTorch website](https://pytorch.org/) for more information. When issues arise concerning the Anonymize software, make sure that no prior version is installed (```$ pip uninstall anonymize_UU``` and/or ```$ pip uninstall anonymoUUs```).

### Download DDP

To download your Instagram data package:

1. Go to www.instagram.com and log in
2. Click on your profile picture, go to *Settings* and *Privacy and Security*
3. Scroll to *Data download* and click *Request download*
4. Enter your email adress and click *Next*
5. Enter your password and click *Request download*

Instagram will deliver your data in a compressed zip folder with format **username_YYYYMMDD.zip** (i.e., Instagram handle and date of download). For Mac users this might be different, so make sure to check that all provided files are zipped into one folder with the name **username_YYYYMMDD.zip**. Save the DDP(s) in the [data folder](/data).

### Create additional files

Before you can run the software, you need to make sure that the [src folder](/src) contains the following items:
* **Facial blurring software**: The *frozen_east_text_detection.pb* software, necessary for the facial blurring of images and videos, can be downloaded from [GitHub](https://github.com/oyyd/frozen_east_text_detection.pb) 
* **Participant file**\*: An overview of all participants' usernames and participant IDs (e.g., participants.csv). We recommend placing this file in the `src` folder. However, you can save this file anywhere you like, as long as you refer to the path correctly while running the software.

**\*** N.B. Only relevant for participant based studies with *predefined* participant IDs. This file can have whatever name you prefer, as long as it is saved as .csv and contains 2 columns; the first being the original instagram handles (e.g., janjansen) and the second the participant IDs (e.g., PP001).

## Run software

When all preceding steps are taken, the data download packages can be pseudonimized. Run the program with (atleast) the arguments `-i` for data input folder (i.e., [data](\data)) and ` -o` data output folder (i.e., [results/output](/results/output)):

```
$ python src/anonymizing_instagram_uu.py [OPTIONS]

Options:
  -i  path to folder containing zipfiles (i.e., -i data/raw)
  -o  path to folder where files will be unpacked and pseudonimized (i.e., -o data/processed)
  -l  path to log file
  -p  path to participants list to use corresponding participant IDs (e.g., -p src/participants.csv)
  -c  replace capitalized names only (when not entering this option, the default = False; not case sensitive) (e.g., -c)

```

An overview of the program's workflow is shown below:
![flowanonymize.png](flowanonymize.png)

The output of the program will be a copy of the zipped data download package with all names, usernames, email addresses, and phone numbers pseudonimized, and all pictures and videos blurred. This pseudonimized data download package is saved in the output folder.

## Evaluation
The evaluation of the performance of the anonymization procedure is described [here](/src/evaluation)
