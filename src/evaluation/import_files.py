from pathlib import Path
import pandas as pd
from zipfile import ZipFile
import shutil
import logging
import argparse
import json
import csv


class ImportFiles:
    """ Import raw, de-identified, and labeled Instagram DDPs """

    def __init__(self, input_folder: Path, results_folder: Path, processed_folder: Path, keys_folder: Path):

        self.logger = logging.getLogger('validating.importing')

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

    def load_results(self, package):
        """ Load Label-Studio results - i.e., (labeled) raw data packages - into dataframe """

        # Load results (json format) of label-studio
        with self.results_folder.open(encoding="utf-8-sig") as f:
            label_results = json.load(f)

        # Create dataframe with most important results: labeled text, type of label, file and package
        labeled_df = pd.DataFrame()
        raw_text = pd.DataFrame()
        for an in range(len(label_results)):
            if label_results[an]['data']['package'] == package:
                file = label_results[an]['data']['file'] + '.json'
                text = label_results[an]['data']['text']
                raw = {'file': file, 'package': package, 'raw_text': text}
                raw_text = raw_text.append(raw, ignore_index=True)

                for i in label_results[an]['completions']:
                    for j in i['result']:
                        if len(j['value']['text']) > 2:
                            inputs = {'labeled_text': [j['value']['text'].strip()], 'label': [j['value']['labels'][0]],
                                      'file': [file], 'package': [package]}
                            inputs = pd.DataFrame(inputs)
                            labeled_df = labeled_df.append(inputs, ignore_index=True)

        return raw_text, labeled_df

    def open_package(self, package, package_hashed):
        """ Load and merge de-identified data packages into a dataframe"""

        anon_text = pd.DataFrame()
        folder = self.processed_folder / package_hashed

        files = list(folder.glob('*.json'))
        for file in files:
            try:
                with open(file, 'r', encoding='utf8') as f2:
                    data = f2.read()
                    input_data = {'text': [data], 'file': [file.name], 'package': [package]}
                    df = pd.DataFrame(input_data)
                    anon_text = anon_text.append(df, ignore_index=True)
            except:
                self.logger.warning(f'      {file} of package {package} is not added')

        return anon_text

    def create_input_label(self):
        """ Creating input file (dataframe with raw files per DDP) for Label-Studio"""

        self.unzipping()
        text = self.merge()

        text_file = self.results_folder / 'text_packages.csv'
        text.to_csv(text_file, index=False)
        self.logger.info(f'Saving merged file: {text_file}')

    def unzipping(self):
        """ Unzip raw DDPs"""

        output = self.results_folder.parent / 'temporary'
        output.mkdir(parents=True, exist_ok=True)

        packages = list(self.input_folder.glob('*.zip'))

        for package in packages:
            with ZipFile(package, 'r') as zipObj:
                self.logger.info(f' Unzipping {package}...')
                zipObj.extractall(output / package)

    def merge(self):
        """ Merge all unpacked files in one file (which forms the input for Label-Studio) """

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

        shutil.rmtree(output)

        return text


def main():
    parser = argparse.ArgumentParser(description='Load static files; raw DDPs, Label-Studio result, key_files.')
    parser.add_argument("--results_folder", "-r", help="Path to ground truth (Label-Studio's result.json)",
                        default=".")
    parser.add_argument("--input_folder", "-i", help="Path to raw DDPs",
                        default=".")
    parser.add_argument("--processed_folder", "-p", help="Path to de-identified DDPs",
                        default=".")
    parser.add_argument("--keys_folder", "-k", help="Path to key files of de-identified DDPs",
                        default=".")
    parser.add_argument("--log_file", "-l", help="Path to log file",
                        default="log_importing_files.txt")


if __name__ == '__main__':
    main()
