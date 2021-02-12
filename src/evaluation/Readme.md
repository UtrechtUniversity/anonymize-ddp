Prerequisites

# Evaluation

Evaluation script for the anonymization script
For this evaluation an example data set which includes:
* A set of DDPs with nonsense content
* A file with results of manually labeling the PII in these DDPs

The example data is available at ...

### Prerequisites

scikit-learn

## Preparatory steps

Before running the software, the following steps need to be taken:

1. **Anonymize DDPs**
2. **[Ground truth data](#groundtruth-data)**

### Anonymize DDPs

Folders with 
* original DDPs
* anonymized DDPs
* key files

### Ground truth data

Download results of manual labeling

The original, non-anonymized text of the DDPs is compared to the labeled ground truth to evaluate labeling process
Create text_packages.csv which includes text of all original DDPs

```
$ cd src/evaluation
$ python merge_inputfiles.py  [OPTIONS]

Options:
  -i path to folder with original DDPs
  -r  path to folder with results of manual labeling

```

## Run softwareto the 

When all preceding steps are taken, the evaluation can be performed. Run the program with (at least) the arguments `...

```
$ cd src/evaluation
$ python validation_script.py [OPTIONS]

Options:
  -r  path to file with results of manual labeling
  -p  path to folder with anonymized datapackages; output of anonymization
  -k  path to folder with key files; output of anonymization

```
## Output
Evaluation metrics:
* table with recall, precision en f1
* four folders with specific occurences of FP, FN, TP and special hashes