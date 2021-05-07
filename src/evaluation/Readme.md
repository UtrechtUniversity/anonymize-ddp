# Evaluation

The evaluation procedure determines the performance of the anonymization code.
It compares results of the automated anonymization with the ideal expected result, i.e., a manually created ground-truth.

For this evaluation an example data set is used which includes:
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
