from zipfile import ZipFile
from pathlib import Path
import logging
import shutil
import pandas as pd
import json
import argparse
import logging


class MergeInputFiles:
    """ Merge all input data packages for labeling """

    def __init__(self, input_folder: Path, results_folder: Path):

        self.logger = logging.getLogger('merging')

        self.input_folder = input_folder
        self.results_folder = results_folder


    def unzipping(self):
        """ Unzip data packages """

        input = self.input_folder
        output = self.results_folder / 'uitgepakt'
        output.mkdir(parents=True, exist_ok=True)

        packages = list(input.glob('*.zip'))

        for package in packages:
            with ZipFile(package, 'r') as zipObj:
                self.logger.info(f' Unzipping {package}...')
                zipObj.extractall(output / package.stem)


    def merge(self):
        """ Merge all unpacked files togher in one dataframe """

        output = self.results_folder / 'uitgepakt'
        text = pd.DataFrame()

        remove = ['autofill.json', 'uploaded_contacts.json', 'account_history.json', 'devices.json', 'information_about_you.json']

        packages = list(output.glob("*"))
        for package in packages:
            files = package.glob('*.json')
            for file in files:
                if file.name not in remove:
                    with file.open(encoding="utf-8-sig") as f:
                        data = json.load(f)
                        input_data = {'text': [data], 'file': [file.stem], 'package': [package.name]}
                        df = pd.DataFrame(input_data)
                        text = text.append(df, ignore_index=True)

        text_file = self.results_folder / 'text_packages.csv'
        text.to_csv(text_file, index=False)
        self.logger.info(f'Saving merged file: {text_file}')

        shutil.rmtree(output)


def init_logging(log_file: Path):
    """
    Initialise Python logger
    :param log_file: Path to the log file.
    """
    logger = logging.getLogger('merging')
    logger.setLevel('INFO')

    # creating a formatter
    formatter = logging.Formatter('- %(name)s - %(levelname)-8s: %(message)s')

    # set up logging to file
    fh = logging.FileHandler(log_file, 'w', 'utf-8')
    fh.setLevel(logging.INFO)

    # Set up logging to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Set handler format
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handler to the root logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def main():
    parser = argparse.ArgumentParser(description='Merge all raw files into one dataframe.')
    parser.add_argument("--input_folder", "-i", help="Enter path to folder where the raw zipped data packages can be found",
                        default=".")
    parser.add_argument("--results_folder", "-r", help="Enter path to folder where result of Label-Studio will be placed",
                        default=".")
    parser.add_argument("--log_file", "-l", help="Enter path to log file",
                        default="log_merge_insta.txt")

    args = parser.parse_args()

    logger = init_logging(Path(args.log_file))
    logger.info(f"Started merging process...")

    mergefiles = MergeInputFiles(Path(args.input_folder), Path(args.results_folder))
    mergefiles.unzipping()
    mergefiles.merge()

    logger.info(f"Finished! :) ")


if __name__ == '__main__':
    main()