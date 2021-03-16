from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
import argparse
from validation_packagebased import ValidateAnonymizationDDP


class ValidateAnonymization:
    """ Detect and anonymize personal information in Instagram data packages"""

    def __init__(self, input_folder: Path, results_folder: Path, processed_folder: Path, keys_folder: Path,
                 labels):

        self.logger = logging.getLogger('validating')
        self.input_folder = Path(input_folder)
        self.results_folder = Path(results_folder)
        self.processed_folder = Path(processed_folder)
        self.keys_folder = Path(keys_folder)

        self.labels = labels

    def statistics(self, check):
        """" Filter data as TPs, FPs, FNs and suspiciously hashed info """

        check['total'] = check['count_hashed_anon'] - check['count_raw']
        FN = check[check['total'] < 0].drop('total', 1)
        FP = check[check['total'] > 0].drop('total', 1)
        TP = check[(check['total'] == 0) & (check['count_anon'] == 0)].drop('total', 1)
        other = check.loc[~check.index.isin(list(FN.index) + list(FP.index) + list(TP.index))].reset_index(drop=True)

        TP = TP.append(FN[FN['count_anon'] == 0], ignore_index=True)
        TP = TP.append(FP[FP['count_raw'] > FP['count_anon']], ignore_index=True)
        FN = FN.append(FP[FP['count_anon'] > 0], ignore_index=True)

        FN['total'] = FN['count_anon']
        FP['total'] = FP['count_hashed_anon'] - FP['count_raw'] - FP['count_anon']
        TP['total'] = TP['count_raw'] - TP['count_anon']
        other['total'] = other['count_raw']

        FN = FN.reset_index(drop=True)
        FP = FP.reset_index(drop=True)
        TP = TP.reset_index(drop=True)

        return TP, FN, FP

    def validation(self, check, TP, FN, FP):
        """ Create dataframe with statistics to evaluate the efficiency of the anonymization process """

        # Count occurrences per label per file
        validation_outcome = check.groupby(['label', 'file'])['count_raw'].sum().reset_index(name='total')
        dataframes = {'TP': TP, 'FN': FN, 'FP': FP}

        for type in dataframes.keys():
            df = dataframes[type]
            df_grouped = df.groupby(['label', 'file'])['total'].sum().reset_index(name=type)
            validation_outcome = validation_outcome.merge(df_grouped, how='outer')

        validation_outcome = validation_outcome.reindex(columns=validation_outcome.columns.tolist() +
                                                                ['Recall', 'Precision', 'F1']).set_index('label')

        # Calculate recall, precision and F1 score per label per file
        final = pd.DataFrame()
        for label in ['DDP_id'] + self.labels:
            try:
                if validation_outcome.index.tolist().count(label) == 1:
                    data = pd.DataFrame(validation_outcome.loc[label]).T
                    data = data.reset_index()
                    data = data.rename(columns={'index': 'label'})
                else:
                    data = validation_outcome.loc[label].reset_index()
                data = data.fillna(0)

                for row, file in enumerate(data['file'].tolist()):
                    data.loc[row, 'Recall'] = self.scikit_input(check, label, file)[0]
                    data.loc[row, 'Precision'] = self.scikit_input(check, label, file)[1]
                    data.loc[row, 'F1'] = self.scikit_input(check, label, file)[2]

                new_row = [label, 'total'] + list(data.sum(numeric_only=True))[:-3] + self.scikit_input(check, label)
                new = pd.DataFrame([new_row], columns=list(data.columns))
                data = data.append(new, ignore_index=True)

                final = final.append(data, ignore_index=True)
            except ValueError:
                next

        final = final.set_index(['label'])
        self.count_files(final)

        return final

    def scikit_input(self, check, label, file=None):
        """ Calculate recall, precision, F1-score """

        if file is None:
            original = list(check['count_raw'][check['label'] == label])
            anonymized = list(check['count_hashed_anon'][check['label'] == label])

        else:
            original = list(check['count_raw'][(check['file'] == file) &
                                               (check['label'] == label)])
            anonymized = list(check['count_hashed_anon'][(check['file'] == file) &
                                                         (check['label'] == label)])

            if len(original) == 0 and (label != 'Username' and label != 'Name'):
                original = [1] * int(original[-1])
                anonymized = [1] * int(anonymized[-1])
                if len(original) < len(anonymized):
                    anonymized[len(original) - 1] = sum(anonymized[len(original) - 1:])
                    anonymized = anonymized[:len(original)]
                elif len(original) > len(anonymized):
                    anonymized.extend([0] * (len(original) - len(anonymized)))

        recall = recall_score(original, anonymized, average='weighted', zero_division=0, sample_weight=original)
        precision = precision_score(original, anonymized, average='weighted', zero_division=0, sample_weight=original)
        f1 = f1_score(original, anonymized, average='weighted', zero_division=0, sample_weight=original)

        return [recall, precision, f1]

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

    # Count labels per DDP and merge final results
    validatingddp = ValidateAnonymizationDDP(args.input_folder, args.results_folder,
                                             args.processed_folder, args.keys_folder)
    df_outcome, labels = validatingddp.merge_packages()

    # Calculate TP, FP, FN and recall, precision and F1-sores
    evalanonym = ValidateAnonymization(args.input_folder, args.results_folder,
                                       args.processed_folder, args.keys_folder, labels)

    # TP, FP, FN per file per label
    path = Path(args.results_folder).parent / 'statistics'
    path.mkdir(parents=True, exist_ok=True)

    TP, FN, FP = evalanonym.statistics(df_outcome)
    logger.info(f"     Saving statistics (TP, FP, FN) to {path}")

    TP.to_csv(path / 'TP.csv', index=False)
    FN.to_csv(path / 'FN.csv', index=False)
    FP.to_csv(path / 'FP.csv', index=False)
    df_outcome.to_csv(path / 'everything.csv', index=False)

    # Validation outcome per file per label
    validation_outcome = evalanonym.validation(df_outcome, TP, FN, FP)
    logger.info(f"     Saving outcome of validation process to {path.parent}")
    validation_outcome.to_csv(path.parent / 'validation_deidentification.csv', index=True)

    logger.info(f"Finished! :) ")


if __name__ == '__main__':
    main()