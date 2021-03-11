from pathlib import Path
import pandas as pd
from zipfile import ZipFile
import shutil
import logging
import argparse
import json
import csv


class ImportFiles:
    """ Detect and anonymize personal information in Instagram data packages"""

    def __init__(self, input_folder: Path, results_folder: Path, processed_folder: Path, keys_folder: Path):

        self.logger = logging.getLogger('validating.import_files')

        self.input_folder = Path(input_folder)
        self.results_folder = Path(results_folder)
        self.processed_folder = Path(processed_folder)
        self.keys_folder = Path(keys_folder)

    def load_keys(self) -> dict:
        """ Load and merge all key files (created in the de-identification process) """

        key_files = list(self.keys_folder.glob('*'))
        all_keys = {}

        for i in key_files:
            keys = {}
            with open(i, 'r') as data:
                for line in csv.reader(data):
                    keys.update({line[0]: line[1]})
                    if line[1] == i.stem.split('keys_')[1]:
                        username = line[0]
            all_keys.update({username: keys})

        return all_keys

    def load_results(self):
        """ Load Label-Studio results (labeled raw data packages) into dataframe """

        # Load results (json format) of label-studio
        with self.results_folder.open(encoding="utf-8-sig") as f:
            label_results = json.load(f)

        # Create dataframe with most important results: labeled text, type of label, file and package
        labeled_df = pd.DataFrame()
        for an in range(len(label_results)):
            file = label_results[an]['data']['file'] + '.json'
            package = label_results[an]['data']['package']
            for i in label_results[an]['completions']:
                for j in i['result']:
                    if len(j['value']['text']) > 2:
                        inputs = {'labeled_text': [j['value']['text'].strip()], 'label': [j['value']['labels'][0]],
                                  'file': [file], 'package': [package]}
                        inputs = pd.DataFrame(inputs)
                        labeled_df = labeled_df.append(inputs, ignore_index=True)

        return labeled_df

    def load_raw_text(self, package):
        """ Load the raw file used in Label-Studio for labeling """

        self.unzipping(package)
        raw_text = self.merge()

        return raw_text

    def unzipping(self, package):
        """ Unzip data packages """

        output = self.results_folder.parent / 'temporary'
        output.mkdir(parents=True, exist_ok=True)

        unpack = list(self.input_folder.glob(f'*{package}.zip'))[0]

        with ZipFile(unpack, 'r') as zipObj:
            self.logger.info(f' Unzipping {package}...')
            zipObj.extractall(output / package)

    def merge(self):
        """ Merge all unpacked files togher in one dataframe """

        output = self.results_folder.parent / 'temporary'
        text = pd.DataFrame()

        remove = ['autofill.json', 'uploaded_contacts.json', 'account_history.json',
                  'devices.json', 'information_about_you.json']

        packages = list(output.glob("*"))
        for package in packages:
            files = package.glob('*.json')
            for file in files:
                if file.name not in remove:
                    with file.open(encoding="utf-8-sig") as f:
                        data = json.load(f)
                        input_data = {'text': [data], 'file': [file.name], 'package': [package.name]}
                        df = pd.DataFrame(input_data)
                        text = text.append(df, ignore_index=True)

        # text_file = self.results_folder / 'text_packages.csv'
        # text.to_csv(text_file, index=False)
        # self.logger.info(f'Saving merged file: {text_file}')

        shutil.rmtree(output)

        return text


def main():
    parser = argparse.ArgumentParser(description='Load static files; raw DDPs, Label-Studio result, key_files.')
    parser.add_argument("--results_folder", "-r", help="Enter path to folder where result of Label-Studio can be found",
                        default=".")
    parser.add_argument("--input_folder", "-i", help="Enter path to folder where the raw data packages can be found",
                        default=".")
    parser.add_argument("--processed_folder", "-p", help="Enter path to folder where the processed (i.e., de-identified) data packages can be found",
                        default=".")
    parser.add_argument("--keys_folder", "-k", help="Enter path to folder where the key files can be found",
                        default=".")
    parser.add_argument("--log_file", "-l", help="Enter path to log file",
                        default="log_importing_files.txt")

if __name__ == '__main__':
    main()