from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
import argparse
from validation_packagebased import ValidateAnonymizationDDP
from import_files import ImportFiles


class ValidateAnonymization:
    """ Detect and anonymize personal information in Instagram data packages"""

    def __init__(self, input_folder: Path, results_folder: Path, processed_folder: Path, keys_folder: Path):

        self.logger = logging.getLogger('validating')
        self.input_folder = Path(input_folder)
        self.results_folder = Path(results_folder)
        self.processed_folder = Path(processed_folder)
        self.keys_folder = Path(keys_folder)

    def statistics(self, df_merged_ddps):
        """" Filter data as TPs, FPs, FNs and suspiciously hashed info """

        df_merged_ddps['total'] = df_merged_ddps['count_hashed_anon'] - df_merged_ddps['count_raw']
        FN = df_merged_ddps[(df_merged_ddps['total'] < 0) & (df_merged_ddps['count_anon'] > 0)].drop('total', 1)
        FP = df_merged_ddps[df_merged_ddps['total'] > 0].drop('total', 1)
        TP = df_merged_ddps[(df_merged_ddps['total'] == 0) & (df_merged_ddps['count_anon'] == 0)].drop('total', 1)
        other = df_merged_ddps.loc[~df_merged_ddps.index.isin(list(FN.index) + list(FP.index) +
                                                              list(TP.index))].reset_index(drop=True)

        TP = TP.append(FN[FN['count_anon'] != FN['count_raw']], ignore_index=True)
        TP = TP.append(other[other['count_anon'] == 0], ignore_index=True)
        TP = TP.append(FP[FP['count_raw'] > FP['count_anon']], ignore_index=True)
        FN = FN.append(FP[FP['count_anon'] > 0], ignore_index=True)

        FN['total'] = FN['count_anon']
        FP['total'] = FP['count_hashed_anon'] - (FP['count_raw'] - FP['count_anon'])
        TP['total'] = TP['count_raw'] - TP['count_anon']
        other['total'] = other['count_raw']

        FN = FN.reset_index(drop=True)
        FP = FP.reset_index(drop=True)
        TP = TP.reset_index(drop=True)

        return TP, FN, FP

    def validation(self, df_merged_ddps, TP, FN, FP):
        """ Create dataframe with statistics to evaluate the efficiency of the anonymization process """

        # Count occurrences per label per file
        validation_outcome = df_merged_ddps.groupby(['label', 'file'])['count_raw'].sum().reset_index(name='total')
        dataframes = {'TP': TP, 'FN': FN, 'FP': FP}

        for type in dataframes.keys():
            df = dataframes[type]
            df_grouped = df.groupby(['label', 'file'])['total'].sum().reset_index(name=type)
            validation_outcome = validation_outcome.merge(df_grouped, how='outer')

        validation_outcome = validation_outcome.reindex(columns=validation_outcome.columns.tolist() +
                                                                ['Recall', 'Precision', 'F1']).set_index('label')

        # Calculate recall, precision and F1 score per label per file
        final = pd.DataFrame()
        for label in validation_outcome.index.unique():
            try:
                if validation_outcome.index.tolist().count(label) == 1:
                    data = pd.DataFrame(validation_outcome.loc[label]).T
                    data = data.reset_index()
                    data = data.rename(columns={'index': 'label'})
                else:
                    data = validation_outcome.loc[label].reset_index()
                data = data.fillna(0)

                for row, file in enumerate(data['file'].tolist()):
                    data.loc[row, ['Recall', 'Precision', 'F1']] = self.validation_scores(df_merged_ddps, label, file)

                new_row = [label, 'total'] + list(data.sum(
                                             numeric_only=True))[:-3] + self.validation_scores(df_merged_ddps, label)
                new = pd.DataFrame([new_row], columns=list(data.columns))
                data = data.append(new, ignore_index=True)
                final = final.append(data, ignore_index=True)

            except ValueError:
                next

        final = final.set_index(['label'])
        self.count_files(final)

        return final

    def validation_scores(self, df_merged_ddps, label, file=None):
        """ Calculate recall, precision, F1-score """

        if file is None:
            original = list(df_merged_ddps['count_raw'][df_merged_ddps['label'] == label])
            anonymized, missed = self.count_anonymized(df_merged_ddps, label)
        else:
            original = list(df_merged_ddps['count_raw'][(df_merged_ddps['file'] == file) &
                            (df_merged_ddps['label'] == label)])
            anonymized, missed = self.count_anonymized(df_merged_ddps, label, file)

        recall, precision, f1 = self.calc_sores(label, original, anonymized, missed)

        return [recall, precision, f1]

    def count_anonymized(self, df_merged_ddps, label, file=None):
        """ Extract the number of correctly(!) hashed items """

        if file is None:
            df_merged = df_merged_ddps[df_merged_ddps['label'] == label].reset_index(drop=True)
        else:
            df_merged = df_merged_ddps[(df_merged_ddps['file'] == file) &
                                       (df_merged_ddps['label'] == label)].reset_index(drop=True)

        df_merged['total_missed'] = 0
        for i in range(len(df_merged)):
            if (df_merged.loc[i, 'count_anon'] == 0) and (df_merged.loc[i, 'count_raw'] >= df_merged.loc[i, 'count_hashed_anon']):
                df_merged.loc[i, 'total_hashed'] = df_merged.loc[i, 'count_raw']
            elif (df_merged.loc[i, 'count_anon'] == 0) and (df_merged.loc[i, 'count_raw'] < df_merged.loc[i, 'count_hashed_anon']):
                df_merged.loc[i, 'total_hashed'] = df_merged.loc[i, 'count_hashed_anon']
            elif df_merged.loc[i, 'count_anon'] > 0:
                df_merged.loc[i, 'total_hashed'] = df_merged.loc[i, 'count_hashed_anon']
                df_merged.loc[i, 'total_missed'] = df_merged.loc[i, 'count_anon']

        anonymized = df_merged['total_hashed'].to_list()
        missed = df_merged['total_missed'].to_list()

        return anonymized, missed

    def calc_sores(self, label, original, anonymized, missed):
        """ Split summed count (__url, __phonenumber, __emailaddress, DDP_id) in list """

        if label != 'Username' and label != 'Name':
            original = [1] * int(sum(original) - sum(missed))
            anonymized = [1] * int(sum(anonymized))
            if len(original) < len(anonymized):
                original.extend([0] * (len(anonymized) - len(original)))
            elif len(original) > len(anonymized):
                anonymized.extend([0] * (len(original) - len(anonymized)))

            original = original + [1] * int(sum(missed))
            anonymized = anonymized + [0] * int(sum(missed))

            recall = recall_score(original, anonymized, average='binary', zero_division=0)
            precision = precision_score(original, anonymized, average='binary', zero_division=0)
            f1 = f1_score(original, anonymized, average='binary', zero_division=0)
        else:
            recall = recall_score(original, anonymized, average='weighted', zero_division=0, sample_weight=original)
            precision = precision_score(original, anonymized, average='weighted', zero_division=0, sample_weight=original)
            f1 = f1_score(original, anonymized, average='weighted', zero_division=0, sample_weight=original)

        return [recall, precision, f1]

    def count_files(self, final):
        """ Create frequency table of labeled raw text per file per package """

        files = sorted(final['file'].unique())
        count_files = pd.DataFrame({'file': files})

        for label in final.index.unique():
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
    parser.add_argument("--results_folder", "-r", help="Enter path to Label-Studio output",
                        default=".")
    parser.add_argument("--input_folder", "-i", help="Enter path to raw DDPs",
                        default=".")
    parser.add_argument("--processed_folder", "-p", help="Enter path to de-identified DDPs",
                        default=".")
    parser.add_argument("--keys_folder", "-k", help="Enter path to folder where the key files can be found",
                        default=".")
    parser.add_argument("--log_file", "-l", help="Enter path to log file",
                        default="log_eval_insta.txt")

    args = parser.parse_args()

    logger = init_logging(Path(args.log_file))
    logger.info(f"Started validation process:")

    # Count labels per DDP and merge final results
    importing = ImportFiles(args.input_folder, args.results_folder, args.processed_folder, args.keys_folder)
    key_files = importing.load_keys()
    packages = list(key_files.keys()) # enter what packages you want to check

    # Count number of labels per file per DDP
    df_outcome = pd.DataFrame()
    number = 1
    for package in packages:
        validatingddp = ValidateAnonymizationDDP(package,args.input_folder, args.results_folder,
                                             args.processed_folder, args.keys_folder)
        data_outcome = validatingddp.count_labels()
        df_outcome = df_outcome.append(data_outcome, ignore_index=True)
        number += 1

    #df_merged_ddps = validatingddp.merge_packages()
    path = Path(args.results_folder).parent / 'statistics'
    path.mkdir(parents=True, exist_ok=True)
    df_outcome.to_csv(path / 'everything_MdV.csv', index=False)

    # Calculate TP, FP, FN and recall, precision and F1-sores
    evalanonym = ValidateAnonymization(args.input_folder, args.results_folder,
                                       args.processed_folder, args.keys_folder)

    # TP, FP, FN per file per label
    TP, FN, FP = evalanonym.statistics(df_outcome)
    logger.info(f"     Saving statistics (TP, FP, FN) to {path}")
    TP.to_csv(path / 'TP.csv', index=False)
    FN.to_csv(path / 'FN.csv', index=False)
    FP.to_csv(path / 'FP.csv', index=False)

    # # Validation outcome per file per label
    validation_outcome = evalanonym.validation(df_outcome, TP, FN, FP)
    logger.info(f"     Saving outcome of validation process to {path.parent}")
    validation_outcome.to_csv(path.parent / 'validation_deidentification.csv', index=True)

    logger.info(f"Finished! :) ")


if __name__ == '__main__':
    main()