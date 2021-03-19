from pathlib import Path
import pandas as pd
import logging
import argparse
import re
from import_files import ImportFiles


class ValidateAnonymizationDDP:
    """ Detect and anonymize personal information in Instagram data packages"""

    def __init__(self, package:Path, input_folder: Path, results_folder: Path, processed_folder: Path, keys_folder: Path):

        self.logger = logging.getLogger('validating.DPP-based')
        self.package = package
        self.input_folder = Path(input_folder)
        self.results_folder = Path(results_folder)
        self.processed_folder = Path(processed_folder)
        self.keys_folder = Path(keys_folder)

        self.regex = r"(?:(?<=\\n)|(?<=[\W]))(?:(?<!instagram.com\/)(?<!stories\/)){}(?=[\W])(?![@])"
        self.regex2 = '{}(?=[\W])'

    def execute(self, anon_text, package, package_hashed, key_file, result, raw_file, labels):
        """ Per DDP per file, count original and hashed PII """

        outcome_df = pd.DataFrame()

        for label in labels:
            labeled_text = self.filter_labels(label, result)
            self.logger.info(f'      Counting text labeled as \'{label}\'')

            if label == 'Email':
                subt = '__emailaddress'
                outcome = self.compare_labels(anon_text, labeled_text, subt, package_hashed, raw_file)
            elif label == 'Phone':
                subt = '__phonenumber'
                outcome = self.compare_labels(anon_text, labeled_text, subt, package_hashed, raw_file)
            elif label == 'URL':
                subt = '__url'
                outcome = self.compare_labels(anon_text, labeled_text, subt, package_hashed, raw_file)
            else:
                outcome = self.compare_names(anon_text, labeled_text, package_hashed, key_file, raw_file, package)

            outcome_df = outcome_df.append(outcome, ignore_index=True)

        outcome_df = self.ddp_merge(outcome_df, package_hashed)

        return outcome_df

    def filter_labels(self, label, result):
        """ Create frequency table of labeled raw text per file per package """

        count_labels = result.groupby(['labeled_text', 'file', 'package',
                                       'label']).size().reset_index(name='labeled_count')
        count_labels = count_labels.sort_values(by=['package', 'file'], ascending=True).reset_index(drop=True)

        labeled_text = count_labels.loc[count_labels['label'] == label]
        labeled_text = labeled_text.reset_index(drop=True)

        return labeled_text

    def compare_names(self, anon_text, labeled_text, package_hashed, key_file, raw_file, package):
        """ Create overview of original and hashed occurance of sensitive info """

        # Add new columns to labeled_text
        labeled_text = labeled_text.reindex(columns=labeled_text.columns.tolist() + ['text_hashed', 'package_hashed',
                                                                                     'count_raw', 'count_anon',
                                                                                     'count_hashed_anon'])
        labeled_text['package_hashed'] = package_hashed

        # Count labeled_text and hashed_text occurances in raw and de-identified DDPs
        for row, file in enumerate(labeled_text['file']):
            text = labeled_text['labeled_text'][row]

            try:
                key_file_lower = {i.lower(): v for i, v in key_file.items()}
                subt = key_file_lower[text.lower().strip()]
            except KeyError:
                self.logger.warning(f'          No hash for {text} from {file} in {package}\'s key file')
                subt = '__NOHASH__'
            labeled_text.loc[row, 'text_hashed'] = subt

            raw = raw_file['raw_text'][raw_file['file'] == file].reset_index(drop=True)[0]
            occ_raw = re.findall(self.regex.format(text.lower()), raw.lower())
            labeled_text.loc[row, 'count_raw'] = len(occ_raw)

            anonymized = anon_text['text'][anon_text['file'] == file].reset_index(drop=True)[0]
            occ_text = re.findall(self.regex.format(text.lower()), str(anonymized).lower())
            labeled_text.loc[row, 'count_anon'] = len(occ_text)

            occ_subt = re.findall(self.regex2.format(subt), str(anonymized))
            labeled_text.loc[row, 'count_hashed_anon'] = len(occ_subt)

        return labeled_text

    def compare_labels(self, anon_text, labeled_text, subt, package_hashed, raw_file):
        """ Create overview of original and hashed occurance of sensitive info """

        labeled_text_long = labeled_text

        # Add new columns to labeled_text
        labeled_text = labeled_text.groupby(['label', 'file', 'package'])['labeled_count'].sum().reset_index(name='labeled_count')
        labeled_text = labeled_text.reindex(columns=labeled_text.columns.tolist() + ['text_hashed', 'package_hashed',
                                                                                     'count_raw', 'count_anon', 'count_hashed_anon'])
        labeled_text['package_hashed'] = package_hashed
        labeled_text['text_hashed'] = subt

        # Count labeled_text and hashed_text occurances in raw and de-identified DDPs
        for row, file in enumerate(labeled_text['file']):
            text = list(labeled_text_long['labeled_text'][labeled_text_long['file'] == file])

            raw = raw_file['raw_text'][(raw_file['file'] == file)].reset_index(drop=True)[0]
            anonymized = anon_text['text'][anon_text['file'] == file].reset_index(drop=True)[0]

            occ_raw = occ_text = 0
            for item in text:
                url = 'https:\S*instagram\w*.com\S*'
                if re.match(url, str(item)):
                    pattern = self.regex2.format(url)
                    occ_raw = len(re.findall(pattern, raw))
                    occ_text = len(re.findall(pattern, str(anonymized)))
                else:
                    try:
                        pattern = self.regex2.format(item)
                        res_raw = re.findall(pattern, raw)
                        res_anon = re.findall(pattern, str(anonymized))
                    except re.error:
                        res_raw = re.findall(self.regex2.format(item.split('+')[1]), raw)
                        res_anon = re.findall(self.regex2.format(item.split('+')[1]), str(anonymized))

                    occ_raw = occ_raw + len(res_raw)
                    occ_text = occ_text + len(res_anon)

            labeled_text.loc[row, 'count_raw'] = occ_raw
            labeled_text.loc[row, 'count_anon'] = occ_text

            occ_subt = re.findall(self.regex2.format(subt), str(anonymized))
            labeled_text.loc[row, 'count_hashed_anon'] = len(occ_subt)

        return labeled_text

    def ddp_merge(self, df_outcome, package_hashed):
        """ Merge all occurrences of DDP PII """

        count_hash = df_outcome.groupby(['file', 'package', 'text_hashed'])['text_hashed'].count().reset_index(name='total')
        count_hash_recur = count_hash[(count_hash['total'] > 1) &
                                      (count_hash['text_hashed'] == '__'+package_hashed.rsplit('_', 1)[0])].reset_index(drop=True)

        pii_index = []
        for i in count_hash_recur.index:
            file = count_hash_recur.loc[i, 'file']
            package = count_hash_recur.loc[i, 'package']
            text_hashed = count_hash_recur.loc[i, 'text_hashed']
            pii_index.extend(df_outcome[(df_outcome['file'] == file) & (df_outcome['package'] == package)
                              & (df_outcome['text_hashed'] == text_hashed)].index)

        pii_package_df = df_outcome.loc[pii_index].reset_index(drop=True)
        check = df_outcome.loc[~df_outcome.index.isin(pii_index)].reset_index(drop=True)

        pii_package = pii_package_df.groupby(['package', 'file', 'text_hashed', 'package_hashed']).sum().reset_index()
        pii_package['count_hashed_anon'] = pii_package['count_hashed_anon'] / 2
        pii_package['label'] = 'DDP_id'

        check = check.append(pii_package, ignore_index=True)

        return check

    def count_labels(self):
        """Count labels and corresponding hashes for all files in DDP"""
    
        self.logger.info(f'  Scoring DDP {self.package}')

        importing = ImportFiles(self.input_folder, self.results_folder, self.processed_folder, self.keys_folder)
        raw_file, result = importing.load_results(self.package)

        key_files = importing.load_keys()
        key_file = key_files[self.package]
        package_hashed = key_file[self.package]

        anon_text = importing.open_package(self.package, package_hashed)

        labels = list(result['label'].unique())

        data_outcome = self.execute(anon_text, self.package, package_hashed, key_file, result, raw_file, labels)

        return data_outcome
    
    
    
    # def merge_packages(self):
    #     # Load fixed files one time
    #     importing = ImportFiles(self.input_folder, self.results_folder, self.processed_folder, self.keys_folder)

    #     key_files = importing.load_keys()
    #     packages = list(key_files.keys()) # enter what packages you want to check

    #     # Count number of labels per file per DDP
    #     df_outcome = pd.DataFrame()
    #     number = 1
    #     for package in packages:
    #         self.logger.info(f'  Scoring DDP \'{package}\' ({number}/{len(packages)})')

    #         raw_file, result = importing.load_results(package)

    #         key_file = key_files[package]
    #         package_hashed = key_file[package]

    #         anon_text = importing.open_package(package, package_hashed)

    #         labels = list(result['label'].unique())

    #         data_outcome = self.execute(anon_text, package, package_hashed, key_file, result, raw_file, labels)
    #         df_outcome = df_outcome.append(data_outcome, ignore_index=True)
    #         number += 1

    #     return df_outcome


def main():
    parser = argparse.ArgumentParser(description='DDP based validation anonymization process.')
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

    evalanonym = ValidateAnonymizationDDP(args.input_folder, args.results_folder,
                                          args.processed_folder, args.keys_folder)
    df_outcome, labels = evalanonym.merge_packages()

    path = Path(args.results_folder).parent / 'statistics'
    df_outcome.to_csv(path / 'everything.csv', index=False)


if __name__ == '__main__':
    main()