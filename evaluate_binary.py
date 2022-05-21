import fire
from sklearn.metrics import recall_score, precision_score, f1_score

from evaluate_graded import create_merged_df


def evaluate(preds_binary_path: str = 'data/binary_scores.tsv', gold_scores_path: str = 'data/gold_scores.tsv',
             target_words_path: str = 'data/target_words.txt') -> None:
    merged_df = create_merged_df(preds_binary_path, gold_scores_path, target_words_path)
    metrics = ('change_binary', 'change_binary_gain', 'change_binary_loss')
    for metric in metrics:
        recall = recall_score(merged_df[f'{metric}_gold'], merged_df[f'{metric}_pred'])
        precision = precision_score(merged_df[f'{metric}_gold'], merged_df[f'{metric}_pred'])
        f1 = f1_score(merged_df[f'{metric}_gold'], merged_df[f'{metric}_pred'])
        print(f'{metric}:', f'f1={f1}', f'precision={precision}', f'recall={recall}')


if __name__ == '__main__':
    fire.Fire(evaluate)
