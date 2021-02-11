from pathlib import Path
import pandas as pd
from functools import reduce
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
import argparse
import json
import csv
import re


class ValidateAnonymization:
    """ Detect and anonymize personal information in Instagram data packages"""

    def __init__(self, results_folder: Path, processed_folder: Path, keys_folder: Path):

        self.logger = logging.getLogger('validating')
        self.results_folder = results_folder
        self.processed_folder = processed_folder
        self.keys_folder = keys_folder

        self.all_keys = self.load_keys()
        self.anon_text = self.load_packages()
        self.count_labels = self.count_labels()

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

    def load_packages(self):
        """ Load and merge de-identified data packages into a dataframe"""

        anon_text = pd.DataFrame()
        packages = list(self.processed_folder.glob("*"))

        for package in packages:
            files = package.glob('*.json')
            for file in files:
                try:
                    with open(file, 'r', encoding='utf8') as f2:
                        data = f2.read()
                        input_data = {'text': [data], 'file': [file.name], 'package': [package.name]}
                        df = pd.DataFrame(input_data)
                        anon_text = anon_text.append(df, ignore_index=True)
                except:
                    self.logger.warning(f'      {file} of package {package} is not added')

        return anon_text

    def load_raw_text(self):
        """ Load the raw file used in Label-Studio for labeling """

        raw_text = pd.read_csv(self.results_folder.parent / 'text_packages.csv')
        raw_text['file'] = raw_text['file'] + '.json'

        return raw_text

    def load_results(self):
        """ Load Label-Studio results (labeled raw data packages) into dataframe """

        # Load results (json format) of label-studio
        label_results = self.results_folder
        with label_results.open(encoding="utf-8-sig") as f:
            label_results = json.load(f)

        # Determine length of label_results(i.e., number of labeled documents )
        length = 0
        for item in label_results:
            length += 1

        # Create dataframe with most important results: labeled text, type of label, file and package
        labeled_df = pd.DataFrame()
        for an in range(length):
            file = label_results[an]['data']['file'] + '.json'
            package = label_results[an]['data']['package']
            for i in label_results[an]['completions']:
                for j in i['result']:
                    if len(j['value']['text']) > 2:
                        inputs = {'text': [j['value']['text'].strip()], 'label': [j['value']['labels'][0]],
                                  'file': [file], 'package': [package]}
                        inputs = pd.DataFrame(inputs)
                        labeled_df = labeled_df.append(inputs, ignore_index=True)

        return labeled_df

    def count_files(self):
        """ Create frequency table of labeled raw text per file per package """

        labeled_df = self.load_results()

        count_files = labeled_df.groupby(['label', 'file']).size().reset_index(name='count')
        count_files = count_files.pivot(index='file', columns='label', values='count').reset_index().rename_axis(None,axis=1)
        count_files = count_files[['file', 'Username', 'Name', 'Email', 'Phone', 'URL']]
        count_files = count_files.rename(columns={'Username': 'Usernames', 'Name': 'First names', 'Email': 'E-mail'})

        count_files['Total'] = count_files.sum(1)

        path = Path(self.results_folder.parent, f'descriptives.csv')
        count_files.to_csv(path, index=False)

        return count_files

    def count_labels(self):
        """ Create frequency table of labeled raw text per file per package """

        labeled_df = self.load_results()

        count_labels = labeled_df.groupby(['text', 'file', 'package', 'label']).size().reset_index(name='count')
        count_labels = count_labels.sort_values(by=['package', 'file'], ascending=True).reset_index(drop=True)

        return count_labels

    def filter_label(self, label):
        """ Filter text frequency dataframe per label """

        count_labels = self.count_labels
        labeled_text = count_labels.loc[count_labels['label'] == label]
        labeled_text = labeled_text.reset_index(drop=True)

        return labeled_text

    def compare_names(self, label):
        """ Create overview of original and hashed occurance of sensitive info """
        # Load necessary data
        anon_text = self.anon_text
        all_keys = self.all_keys
        raw_text = self.load_raw_text()
        labeled_text = self.filter_label(label)

        labeled_text = labeled_text.rename(columns={'count': 'labeled_count', 'text': 'labeled_text'})

        labeled_text['text_hashed'] = ''
        labeled_text['package_hashed'] = ''

        labeled_text['count_raw'] = ''
        labeled_text['count_anon'] = ''
        labeled_text['count_hashed_anon'] = ''

        for row in range(labeled_text.shape[0]):
            package = labeled_text['package'][row]
            file = labeled_text['file'][row]
            text = labeled_text['labeled_text'][row]

            try:
                all_keys_lower = {i.lower(): v for i, v in all_keys[package].items()}
                subt = all_keys_lower[text.lower().strip()]
            except KeyError:
                # self.logger.warning(f'     No hash for {text} in {package}\'s key file')
                print(f'     No hash for {text} in {package}\'s key file')
                subt = '__NOHASH__'
            labeled_text.loc[row, 'text_hashed'] = subt

            package_hash = all_keys[package][package]
            labeled_text.loc[row, 'package_hashed'] = package_hash

            raw = raw_text['text'][(raw_text['package'] == package) & (raw_text['file'] == file)].reset_index(drop=True)
            occ_raw = re.findall("(?<!instagram.com\/)(?<=[\s\'\"{@\/\.#])" + text.lower() + "(?=[\W])", str(raw[0]).lower())
            occ_raw_stories = re.findall("(?<=stories\/)(?<=[\s\'\"{@\/\.#])" + text.lower() + "(?=[\W])", str(raw[0]).lower())
            labeled_text.loc[row, 'count_raw'] = len(occ_raw) - len(occ_raw_stories)

            anonymized = anon_text['text'][(anon_text['package'] == package_hash) & (anon_text['file'] == file)].reset_index(drop=True)
            occ_text = re.findall(text.lower() + "(?=[\W])", str(anonymized[0]).lower())
            labeled_text.loc[row, 'count_anon'] = len(occ_text)

            occ_subt = re.findall(subt + "(?=[\W])", str(anonymized[0]))
            labeled_text.loc[row, 'count_hashed_anon'] = len(occ_subt)

        # Save dataframe with all labeled text and its (hashed) occurances in the anonymized packages
        self.logger.info(f'     Save {label}\'s overview to occurances_{label}.csv')
        path = Path(self.results_folder.parent, 'occurances')
        path.mkdir(parents=True, exist_ok=True)
        labeled_text.to_csv(path / f'occurances_{label}.csv', index=False)

        # Evaluation of the efficiency of anonymization
        self.statistics(labeled_text, label)
        data_outcome = self.accuracy(labeled_text, label)

        return data_outcome

    def compare_labels(self, label):
        """ Create overview of original and hashed occurance of sensitive info """

        # Load necessary data
        anon_text = self.anon_text
        all_keys = self.all_keys
        raw_text = self.load_raw_text()
        labeled_text = self.filter_label(label)

        labeled_text_long = labeled_text.rename(columns={'count': 'labeled_count', 'text': 'labeled_text'})

        labeled_text = labeled_text.groupby(['label', 'file', 'package']).size().reset_index(name='count')
        labeled_text = labeled_text.rename(columns={'count': 'labeled_count', 'label': 'labeled_text'})

        labeled_text['text_hashed'] = ''
        labeled_text['package_hashed'] = ''

        labeled_text['count_raw'] = ''
        labeled_text['count_anon'] = ''
        labeled_text['count_hashed_anon'] = ''

        for row in range(labeled_text.shape[0]):
            package = labeled_text['package'][row]
            file = labeled_text['file'][row]
            text = list(labeled_text_long['labeled_text'][(labeled_text_long['package'] == package) &
                                                     (labeled_text_long['file'] == file)])

            if label == 'Email':
                subt = '__emailaddress'
            elif label == 'Phone':
                subt = '__phonenumber'
            else:
                subt = '__url'
            labeled_text.loc[row, 'text_hashed'] = subt

            package_hash = all_keys[package][package]
            labeled_text.loc[row, 'package_hashed'] = package_hash

            raw = raw_text['text'][(raw_text['package'] == package) & (raw_text['file'] == file)].reset_index(
                drop=True)
            anonymized = anon_text['text'][
                (anon_text['package'] == package_hash) & (anon_text['file'] == file)].reset_index(drop=True)

            occ_raw = occ_text = 0
            for item in text:
                try:
                    res_raw = re.findall(item + "(?=[\W])", str(raw[0]))
                    res_anon = re.findall(item + "(?=[\W])", str(anonymized[0]))
                except re.error:
                    res_raw = re.findall(item.split('+')[1] + "(?=[\W])", str(raw[0]))
                    res_anon = re.findall(item.split('+')[1] + "(?=[\W])", str(anonymized[0]))

                occ_raw = occ_raw + len(res_raw)
                occ_text = occ_text + len(res_anon)

            labeled_text.loc[row, 'count_raw'] = occ_raw
            labeled_text.loc[row, 'count_anon'] = occ_text

            occ_subt = re.findall(subt + "(?=[\W])", str(anonymized[0]))
            labeled_text.loc[row, 'count_hashed_anon'] = len(occ_subt)

        # Save dataframe with all labeled text and its (hashed) occurances in the anonymized packages
        self.logger.info(f'     Save {label}\'s overview to occurances_{label}.csv')
        path = Path(self.results_folder.parent, 'occurances')
        path.mkdir(parents=True, exist_ok=True)
        labeled_text.to_csv(path / f'occurances_{label}.csv', index=False)

        # Evaluation of the efficiency of anonymization
        self.statistics(labeled_text, label)
        data_outcome = self.accuracy(labeled_text, label)

        return data_outcome

    def true_positives(self, check):
        """" View the correctly hashed usernames """

        correct = check[(check['count_hashed_anon'] == check['count_raw']) & (check['count_anon'] == 0)]
        correct = correct.reset_index(drop=True)

        return correct, 'true_positives'

    def false_negatives(self, check):
        """ View the false negatives (the usernames that were missed in the anotation process) """

        missed = check[(check['count_anon'] > 0)]
        missed = missed.reset_index(drop=True)

        return missed, 'false_negatives'

    def false_positives(self, check):
        """ View the false positives (the words that were hashed when they shouldn't have) """

        toomuch = check[check['count_hashed_anon'] > check['count_raw']]
        toomuch = toomuch.reset_index(drop=True)

        return toomuch, 'false_positives'

    def suspicious_hashes(self, check):
        """ View words of which both the original and the hash no longer appear in the anonymized packages"""

        weird = check[(check['count_hashed_anon'] < check['count_raw']) & (
                    check['count_anon'] == 0)]
        weird = weird.reset_index(drop=True)

        return weird, 'suspicious_hashes'

    def statistics(self, check, label):
        """" Save the TP, FP, FN and suspiciously hashed info """

        functions = [self.true_positives(check), self.false_negatives(check), self.false_positives(check), self.suspicious_hashes(check)]

        for function in functions:
            statistics = function[0]
            path = Path(self.results_folder.parent, function[1])
            path.mkdir(parents=True, exist_ok=True)
            statistics.to_csv(Path(path, f'{function[1]}_{label}.csv'), index=False)

    def accuracy(self, check, label):
        """ Create dataframe with statistics to evaluate the efficiency of the anonymization process """

        TP = self.true_positives(check)[0].groupby(['file']).size().reset_index(name='TP')
        FN = self.false_negatives(check)[0].groupby(['file']).size().reset_index(name='FN')
        FP = self.false_positives(check)[0].groupby(['file']).size().reset_index(name='FP')
        tot = check.groupby(['file']).size().reset_index(name='total')

        df_outcome = reduce(lambda left, right: pd.merge(left, right, how='outer', on='file'), [TP, FN, FP, tot])

        df_outcome['Recall'] = ''
        df_outcome['Precision'] = ''
        df_outcome['F1'] = ''

        for i in range(df_outcome.shape[0]):
            file = df_outcome['file'][i]

            original = list(check['labeled_count'][check['file'] == file])
            anonymized = list(check['count_hashed_anon'][check['file'] == file])

            df_outcome.loc[i, 'Recall'] = recall_score(original, anonymized, average='weighted', zero_division=0)
            df_outcome.loc[i, 'Precision'] = precision_score(original, anonymized, average='weighted', zero_division=0)
            df_outcome.loc[i, 'F1'] = f1_score(original, anonymized, average='weighted', zero_division=0)

        df_outcome.loc[i+1, 'file'] = 'total'
        df_outcome.loc[i+1, ['TP', 'FN', 'FP', 'total']] = list(df_outcome.sum(numeric_only=True))
        df_outcome.loc[i+1, ['Recall', 'Precision', 'F1']] = list(df_outcome.mean())[-3:]
        df_outcome.insert(0, 'label', label)

        return df_outcome


def init_logging(log_file: Path):
    """
    Initialise Python logger
    :param log_file: Path to the log file.
    """
    logger = logging.getLogger('validating')
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
    parser = argparse.ArgumentParser(description='Validatie anonymization process.')
    parser.add_argument("--results_folder", "-r", help="Enter path to folder where result of Label-Studio can be found",
                        default=".")
    parser.add_argument("--processed_folder", "-p", help="Enter path to folder where the processed (i.e., de-identified) data packages can be found",
                        default=".")
    parser.add_argument("--keys_folder", "-k", help="Enter path to folder where the key files can be found",
                        default=".")
    parser.add_argument("--log_file", "-l", help="Enter path to log file",
                        default="log_eval_insta.txt")

    args = parser.parse_args()

    logger = init_logging(Path(args.log_file))
    logger.info(f"Started validation process:")

    evalanonym = ValidateAnonymization(Path(args.results_folder), Path(args.processed_folder), Path(args.keys_folder))
    evalanonym.count_files()

    labels = ['Username', 'Name', 'Email', 'Phone', 'URL']

    df_outcome = pd.DataFrame()
    for label in labels:
        if label == 'Username' or label == 'Name':
            data_outcome = evalanonym.compare_names(label)
        else:
            data_outcome = evalanonym.compare_labels(label)
        df_outcome = df_outcome.append(data_outcome)

    path = Path(Path(args.results_folder).parent, 'Validation_Outcome.csv')
    logger.info(f"     Saving validation outcome to {path}")
    df_outcome.to_csv(path, index=False)

    logger.info(f"Finished! :) ")

if __name__ == '__main__':
    main()