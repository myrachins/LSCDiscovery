import fire
import pandas as pd
from scipy.stats import spearmanr


def create_merged_df(preds_graded_path: str, gold_scores_path: str, target_words_path: str):
    with open(target_words_path) as f:
        target_words = {word.rstrip() for word in f.readlines()}

    gold_scores_df = pd.read_csv(gold_scores_path, sep='\t')
    preds_df = pd.read_csv(preds_graded_path, sep='\t')
    gold_scores_df = gold_scores_df[gold_scores_df['lemma'].isin(target_words)]
    preds_df = preds_df[preds_df['word'].isin(target_words)]

    merged_df = pd.merge(gold_scores_df, preds_df, how='left', left_on='lemma', right_on='word',
                         suffixes=('_gold', '_pred'))
    return merged_df


def evaluate(preds_graded_path: str = 'data/graded_scores.tsv', gold_scores_path: str = 'data/gold_scores.tsv',
             target_words_path: str = 'data/target_words.txt') -> None:
    merged_df = create_merged_df(preds_graded_path, gold_scores_path, target_words_path)
    change_graded_score = spearmanr(merged_df['change_graded_gold'], merged_df['change_graded_pred'])
    compare_score = spearmanr(merged_df['COMPARE_gold'], merged_df['COMPARE_pred'])
    print('change_graded:', change_graded_score)
    print('COMPARE:', compare_score)


if __name__ == '__main__':
    fire.Fire(evaluate)
