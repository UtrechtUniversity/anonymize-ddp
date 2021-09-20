"""Test script for validation of anonimyzation procedure

The validation scripts evaluate the performance of anonimyze-ddp.
This test script checks whether these validation scripts produce concistent results.
Unpack test_validation_data.zip to obtain the corresponding test data set which contains orginal, anonymized and ground truth data for one DDP

Run this script when you modify the validation procedure.
"""

import pytest
import pandas as pd
from pathlib import Path
from validation.import_files import ImportFiles
from validation.validation_execution import ValidateExecution
from validation.validation_packagebased import ValidatePackage

@pytest.fixture(autouse=True,scope="class")
def im_object():
    """Instantiate ImportFiles class with test data; created object is used in all tests""" 

    data_dir = Path("./test_validation_data")
    processed = data_dir / "processed"
    key_dir = data_dir / "keys"
    res_file = data_dir / "result_horses.json"

    im = ImportFiles(data_dir,res_file,processed,key_dir)

    return im

def test_validation(im_object):
    """Test overall validation procedure for consistency of results""" 

    package = "horsesarecool52_20201020"

    # Load data: raw, anonymized and ground truth
    raw_file, result = im_object.load_results(package)    
    key_files = im_object.load_keys()
    key_file = key_files[package]
    package_hashed = key_file[package]
    anon_text = im_object.open_package(package, package_hashed)

    # Count labels and hashes per file 
    df_outcome = pd.DataFrame()
    val_ddp = ValidatePackage(package, anon_text, package_hashed, key_file, result, raw_file)   
    data_outcome = val_ddp.execute()
    df_outcome = df_outcome.append(data_outcome, ignore_index=True)

    # Calculate total scores 
    val_exc = ValidateExecution(Path("./test"), df_outcome)
    df_val = val_exc.validation()
    
    true_pos = df_val['TP'].values
    # Check totals of true positives
    test_values = [250,2,252,21,1,22,5,223,228,10,10,76,1,77,22,171,219,3,665,610,1,13,1704]
    assert (true_pos == test_values).all()