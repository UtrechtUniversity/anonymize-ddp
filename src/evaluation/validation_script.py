from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
import argparse
import re
from import_files import ImportFiles


class ValidateAnonymization:
    """ Detect and anonymize personal information in Instagram data packages"""

    def __init__(self, input_folder: Path, results_folder: Path, processed_folder: Path, keys_folder: Path,
                 package, package_hashed, key_file, result, raw_file, labels):

        self.logger = logging.getLogger('validating')
        self.input_folder = Path(input_folder)
        self.results_folder = Path(results_folder)
        self.processed_folder = Path(processed_folder)
        self.keys_folder = Path(keys_folder)

        self.package = package
        self.package_hashed = package_hashed
        self.key_file = key_file
        self.result = result
        self.raw_file = raw_file
        self.labels = labels

        self.regex = r"(?:(?<=\\n)|(?<=[\W]))(?:(?<!instagram.com\/)(?<!stories\/)){}(?=[\W])(?![@])"
        self.regex2 = '{}(?=[\W])'

    def open_package(self):
        """ Load and merge de-identified data packages into a dataframe"""

        anon_text = pd.DataFrame()
        folder = self.processed_folder / self.package_hashed

        files = list(folder.glob('*.json'))
        for file in files:
            try:
                with open(file, 'r', encoding='utf8') as f2:
                    data = f2.read()
                    input_data = {'text': [data], 'file': [file.name], 'package': [self.package]}
                    df = pd.DataFrame(input_data)
                    anon_text = anon_text.append(df, ignore_index=True)
            except:
                self.logger.warning(f'      {file} of package {self.package} is not added')

        return anon_text

    def filter_labels(self, label):
        """ Create frequency table of labeled raw text per file per package """

        count_labels = self.result.groupby(['labeled_text', 'file', 'package', 'label']).size().reset_index(name='labeled_count')
        count_labels = count_labels.sort_values(by=['package', 'file'], ascending=True).reset_index(drop=True)

        labeled_text = count_labels.loc[count_labels['label'] == label]
        labeled_text = labeled_text.reset_index(drop=True)

        return labeled_text

    def execute(self):
        """ Per DDP per file, count original and hashed PII """

        anon_text = self.open_package()
        outcome_df = pd.DataFrame()

        for label in self.labels:
            labeled_text = self.filter_labels(label)
            self.logger.info(f'      Counting text labeled as \'{label}\'')

            if label == 'Email':
                subt = '__emailaddress'
                outcome = self.compare_labels(anon_text, labeled_text, subt)
            elif label == 'Phone':
                subt = '__phonenumber'
                outcome = self.compare_labels(anon_text, labeled_text, subt)
            elif label == 'URL':
                subt = '__url'
                outcome = self.compare_labels(anon_text, labeled_text, subt)
            else:
                outcome = self.compare_names(anon_text, labeled_text)

            outcome_df = outcome_df.append(outcome)

        return outcome_df

    def compare_names(self, anon_text, labeled_text):
        """ Create overview of original and hashed occurance of sensitive info """

        # Add new columns to labeled_text
        labeled_text = labeled_text.reindex(columns=labeled_text.columns.tolist() + ['text_hashed', 'package_hashed',
                                                                                     'count_raw', 'count_anon', 'count_hashed_anon'])
        labeled_text['package_hashed'] = self.package_hashed

        # Count labeled_text and hashed_text occurances in raw and de-identified DDPs
        for row, file in enumerate(labeled_text['file']):
            text = labeled_text['labeled_text'][row]

            try:
                key_file_lower = {i.lower(): v for i, v in self.key_file.items()}
                subt = key_file_lower[text.lower().strip()]
            except KeyError:
                self.logger.warning(f'          No hash for {text} from {file} in {self.package}\'s key file')
                subt = '__NOHASH__'
            labeled_text.loc[row, 'text_hashed'] = subt

            raw = self.raw_file['text'][self.raw_file['file'] == file].reset_index(drop=True)
            occ_raw = re.findall(self.regex.format(text.lower()), str(raw[0]).lower())
            labeled_text.loc[row, 'count_raw'] = len(occ_raw)

            anonymized = anon_text['text'][anon_text['file'] == file].reset_index(drop=True)
            occ_text = re.findall(self.regex2.format(text.lower()), str(anonymized[0]).lower())
            labeled_text.loc[row, 'count_anon'] = len(occ_text)

            occ_subt = re.findall(self.regex2.format(subt), str(anonymized[0]))
            labeled_text.loc[row, 'count_hashed_anon'] = len(occ_subt)

        return labeled_text

    def compare_labels(self, anon_text, labeled_text, subt):
        """ Create overview of original and hashed occurance of sensitive info """

        labeled_text_long = labeled_text

        # Add new columns to labeled_text
        labeled_text = labeled_text.groupby(['label', 'file', 'package'])['labeled_count'].sum().reset_index(name='labeled_count')
        labeled_text = labeled_text.reindex(columns=labeled_text.columns.tolist() + ['text_hashed', 'package_hashed',
                                                                                     'count_raw', 'count_anon', 'count_hashed_anon'])
        labeled_text['package_hashed'] = self.package_hashed
        labeled_text['text_hashed'] = subt

        # Count labeled_text and hashed_text occurances in raw and de-identified DDPs
        for row, file in enumerate(labeled_text['file']):
            text = list(labeled_text_long['labeled_text'][labeled_text_long['file'] == file])

            raw = self.raw_file['text'][(self.raw_file['file'] == file)].reset_index(drop=True)
            anonymized = anon_text['text'][anon_text['file'] == file].reset_index(drop=True)

            occ_raw = occ_text = 0
            for item in text:
                url = 'https:\S*instagram\w*.com\S*'
                if re.match(url, str(item)):
                    pattern = url + '(?=[\"\'\.\s,}])'
                    occ_raw = len(re.findall(pattern, str(raw[0])))
                    occ_text = len(re.findall(pattern, str(anonymized[0])))
                else:
                    try:
                        pattern = self.regex2.format(item)
                        res_raw = re.findall(pattern, str(raw[0]))
                        res_anon = re.findall(pattern, str(anonymized[0]))
                    except re.error:
                        res_raw = re.findall(self.regex2.format(item.split('+')[1]), str(raw[0]))
                        res_anon = re.findall(self.regex2.format(item.split('+')[1]), str(anonymized[0]))

                    occ_raw = occ_raw + len(res_raw)
                    occ_text = occ_text + len(res_anon)

            labeled_text.loc[row, 'count_raw'] = occ_raw
            labeled_text.loc[row, 'count_anon'] = occ_text

            occ_subt = re.findall(self.regex2.format(subt), str(anonymized[0]))
            labeled_text.loc[row, 'count_hashed_anon'] = len(occ_subt)

        return labeled_text

    def statistics(self, check):
        """" Filter data as TPs, FPs, FNs and suspiciously hashed info """

        FP = check[check['count_hashed_anon'] > check['count_raw']].reset_index(drop=True)
        FP['total'] = FP['count_hashed_anon'] - FP['count_raw']

        TP = check[(check['count_hashed_anon'] == check['count_raw']) &
                        (check['count_anon'] == 0)].reset_index(drop=True)
        TP = TP.append(FP)
        TP['total'] = TP['count_raw']

        FN = check[(check['count_anon'] > 0)].reset_index(drop=True)
        FN['total'] = FN['count_anon']

        other = check[(check['count_hashed_anon'] < check['count_raw']) &
                      (check['count_anon'] == 0)].reset_index(drop=True)
        other['total'] = other['count_raw']

        return TP, FN, FP, other

    def validation(self, df_outcome, TP, FN, FP, other):
        """ Create dataframe with statistics to evaluate the efficiency of the anonymization process """

        # Count occurences per label per file
        validation_outcome = df_outcome.groupby(['label', 'file'])['count_raw'].sum().reset_index(name='total')
        dataframes = {'TP': TP, 'FN': FN, 'FP': FP, 'other': other}

        for type in dataframes.keys():
            df = dataframes[type]
            df_grouped = df.groupby(['label', 'file'])['total'].sum().reset_index(name=type)
            validation_outcome = validation_outcome.merge(df_grouped, how='outer')

        validation_outcome = validation_outcome.reindex(columns=validation_outcome.columns.tolist() +
                                                                ['Recall', 'Precision', 'F1']).set_index('label')

        # Calculate recall, precision and F1 score per label per file
        final = pd.DataFrame()
        for label in self.labels:
            if validation_outcome.index.tolist().count(label) == 1:
                data = pd.DataFrame(validation_outcome.loc[label]).T
                data = data.reset_index()
                data = data.rename(columns={'index':'label'})
            else:
                data = validation_outcome.loc[label].reset_index()
            data = data.fillna(0)

            for row, file in enumerate(data['file'].tolist()):
                original = list(df_outcome['count_raw'][(df_outcome['file'] == file) &
                                                        (df_outcome['label'] == label)])
                anonymized = list(df_outcome['count_hashed_anon'][(df_outcome['file'] == file) &
                                                        (df_outcome['label'] == label)])

                if label != 'Username' and label != 'Name':
                    original = [1] * int(original[-1])
                    anonymized = [1] * int(anonymized[-1])
                    if len(original) < len(anonymized):
                        anonymized[len(original)-1] = sum(anonymized[len(original)-1:])
                        anonymized = anonymized[:len(original)]
                    elif len(original) > len(anonymized):
                        anonymized.extend([0] * (len(original) - len(anonymized)))

                data.loc[row, 'Recall'] = recall_score(original, anonymized, average='weighted', zero_division=0)
                data.loc[row, 'Precision'] = precision_score(original, anonymized, average='weighted', zero_division=0)
                data.loc[row, 'F1'] = f1_score(original, anonymized, average='weighted', zero_division=0)

            new_row = [label, 'total'] + list(data.sum(numeric_only=True))[:-3] + (list(data.mean())[-3:])
            new = pd.DataFrame([new_row], columns=list(data.columns))
            data = data.append(new, ignore_index=True)

            final = final.append(data)

        final = final.set_index(['label'])
        self.count_files(final)

        return final

    def count_files(self, final):
        """ Create frequency table of labeled raw text per file per package """

        files = sorted(final['file'].unique())
        count_files = pd.DataFrame({'file': files})

        for label in self.labels:
            new = final.loc[label, ['file', 'total']]
            count_files = count_files.merge(new.rename(columns={'total': label}), how='outer')

        path = Path(self.results_folder.parent, f'descriptives.csv')
        count_files.to_csv(path, index=False)

        return count_files


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
    parser.add_argument("--input_folder", "-i", help="Enter path to folder where the raw data packages can be found",
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

    # Load fixed files one time
    importing = ImportFiles(args.input_folder, args.results_folder, args.processed_folder, args.keys_folder)
    results = importing.load_results()
    key_files = importing.load_keys()
    packages = list(key_files.keys()) # enter what packages you want to check
    # packages = ['100billionfaces_20201021']

    # Count number of labels per file per DDP
    df_outcome = pd.DataFrame()
    number = 1
    for package in packages:
        logger.info(f'  Scoring DDP \'{package}\' ({number}/{len(packages)})')

        result = results[results['package'] == package]
        raw_file = importing.load_raw_text(package)

        key_file = key_files[package]
        package_hashed = key_file[package]

        labels = list(result['label'].unique())

        evalanonym = ValidateAnonymization(args.input_folder, args.results_folder,
                                           args.processed_folder, args.keys_folder,
                                           package, package_hashed, key_file, result, raw_file, labels)

        data_outcome = evalanonym.execute()
        df_outcome = df_outcome.append(data_outcome)
        number += 1

    # TP, FP, FN and other per file per DDP
    path = Path(args.results_folder).parent / 'statistics'
    path.mkdir(parents=True, exist_ok=True)

    TP, FN, FP, other = evalanonym.statistics(df_outcome)
    logger.info(f"     Saving statistics (TP, FP, FN, and other) to {path}")

    TP.to_csv(path / 'TP.csv', index=False)
    FN.to_csv(path / 'FN.csv', index=False)
    FP.to_csv(path / 'FP.csv', index=False)
    other.to_csv(path / 'other.csv', index=False)
    df_outcome.to_csv(path / 'everything.csv', index=False)

    # Validation outcome per label per file
    validation_outcome = evalanonym.validation(df_outcome, TP, FN, FP, other)
    logger.info(f"     Saving outcome of validation process to {path.parent}")
    validation_outcome.to_csv(path.parent / 'validation_deidentification.csv', index=True)

    logger.info(f"Finished! :) ")


if __name__ == '__main__':
    main()