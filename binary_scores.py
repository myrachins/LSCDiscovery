import fire
import pandas as pd


def run_binary(graded_scores_path: str = 'data/graded_scores.tsv', output_path: str = 'data/binary_scores.tsv',
               threshold: float = 0.7015):
    data_df = pd.read_csv(graded_scores_path, sep='\t')
    change_mask = (data_df['change_graded'] > threshold)
    data_df['change_binary'] = 0
    data_df.loc[change_mask, 'change_binary'] = 1

    data_df['change_binary_gain'] = data_df['change_binary']
    data_df['change_binary_loss'] = data_df['change_binary']

    data_df.to_csv(output_path, index=False, sep='\t')


if __name__ == '__main__':
    fire.Fire(run_binary)
