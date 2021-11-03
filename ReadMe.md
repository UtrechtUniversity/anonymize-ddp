# Anonymize-DDP

Pseudonimizing software for data download packages (DDP), specifically focussed on Instagram.

## Table of Contents
* [About Anonymize-DDP](#about-anonymize-ddp)
  * [Built with](#built-with)
  * [License](#license)
  * [Attribution and academic use](#attribution-and-academic-use)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Preparatory steps](#preparatory-steps)
    * [Clone repository](#clone-repository)
    * [Download DDP](#download-ddp)
    * [Create additional files](#create-additional-files)
  * [Run software](#run-software)
  * [Validation](#validation)
  
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
The scientific paper detailing the first release of anonymize-ddp is available [here](https://doi.org/10.3233/DS-210035).

A data set consisting of 11 personal Instagram archives, or Data-Download Packages, was created to [validate](/anonymize/validation) the anonymization procedure.
This data set is publicly available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4472606.svg)](https://doi.org/10.5281/zenodo.4472606)


# Getting Started

## Prerequisites
This project makes use of Python 3.8 and [Poetry](https://python-poetry.org/) for managing dependencies. 

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
$ cd anonymize-ddp/anonymize-ddp

# Install dependencies
poetry install 

```

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
* **Participant file**\*: An overview of all participants' usernames and participant IDs (e.g., participants.csv). We recommend placing this file in the `anonymize` folder. However, you can save this file anywhere you like, as long as you refer to the path correctly while running the software.

**\*** N.B. Only relevant for participant based studies with *predefined* participant IDs. This file can have whatever name you prefer, as long as it is saved as .csv and contains 2 columns; the first being the original instagram handles (e.g., janjansen) and the second the participant IDs (e.g., PP001).

## Run software

When all preceding steps are taken, the data download packages can be pseudonimized. 
Note that the `poetry run` command executes the given command inside the project’s virtual environment.
Run the program with (atleast) the arguments `-i` for data input folder (i.e., [data](\data)) and ` -o` data output folder (i.e., [results/output](/results/output)):

```
$ poetry run python anonymize/anonymizing_instagram_uu.py [OPTIONS]

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

## Validation

The validation procedure determines the performance of anonymization code _concerning deidentification of text_.
It compares results of the automated anonymization with the ideal expected result, i.e., a manually created ground-truth.

For this validation an example data set is used which includes:
* A set of 11 DDPs with nonsense content
* A groundtruth file with results of manually labeling the PII in these DDPs

The example data set is available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4472606.svg)](https://doi.org/10.5281/zenodo.4472606).

### Prerequisites

scikit-learn

## Preparatory steps

Before running the software, the following steps need to be taken:

1. **Collect data: both Instagram DDPs and corresponding ground truth data**
2. **[Anonymize Instagram DDPs](#anonymize-ddps)**

### Anonymize DDPs
Run the automated anonymization as described in the [main Readme](/../../)
After the anonymization, make sure you have separate folders with the following data:
* original DDPs
* anonymized DDPs
* key files


## Perform evaluation

When all preceding steps are taken, the evaluation can be performed. 

```
$ cd anonymize/validation
$ poetry run python validation_script.py [OPTIONS]

Options:
  -r  path to file with results of manual labeling
  -p  path to folder with anonymized datapackages; output of anonymization
  -k  path to folder with key files; output of anonymization

```
## Output
Evaluation metrics:
* table with recall, precision en f1
* four folders with specific occurences of FP, FN, TP and special hashes

## Testing
Run tests with available test data to check the consistency of the evaluation procedure 
From the root folder:

```
# Go to the main folder of the poetry project
$ cd anonymize-ddp/anonymize-ddp

# Run the test 
$ poetry run pytest

```