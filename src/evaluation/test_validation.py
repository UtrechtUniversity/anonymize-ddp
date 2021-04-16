import pytest
from pathlib import Path
from import_files import ImportFiles


@pytest.fixture(autouse=True,scope="class")
def im_object():
    """Instantiate ImportFiles class with test data; created object is used in all tests""" 

    data_dir = Path("../../data/test_data")
    processed = data_dir / "processed"
    key_dir = data_dir / "keys"
    res_file = data_dir / "labeled.json"

    im = ImportFiles(data_dir,res_file,processed,key_dir)

    return im

def test_load_keys(im_object):

    all_keys = im_object.load_keys()
    key_dict = all_keys['insta4dummy_20201105']

    assert len(key_dict.keys()) == 9126