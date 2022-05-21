import json
import typing as tp
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import fire
from tqdm import tqdm


def get_files_num(dir_path: str):
    return len(list(Path(dir_path).iterdir()))


class Predictor(ABC):
    @abstractmethod
    def predict(self, out_vector_1, out_vector_2):
        pass


class VectorsDotPredictor(Predictor):
    def __init__(self, normalize: bool = True, norm_ord: int = 2):
        self.normalize = normalize
        self.norm_ord = norm_ord

    def predict(self, out_vector_1, out_vector_2):
        out_vector_1 = np.array(out_vector_1)
        out_vector_2 = np.array(out_vector_2)

        if self.normalize:
            out_vector_1 /= np.linalg.norm(out_vector_1, ord=self.norm_ord)
            out_vector_2 /= np.linalg.norm(out_vector_2, ord=self.norm_ord)

        return np.sum(out_vector_1 * out_vector_2)


class VectorsDistPredictor(Predictor):
    def __init__(self, normalize: bool = True, norm_ord: int = 2):
        self.normalize = normalize
        self.norm_ord = norm_ord

    def predict(self, out_vector_1, out_vector_2):
        out_vector_1 = np.array(out_vector_1)
        out_vector_2 = np.array(out_vector_2)

        if self.normalize:
            out_vector_1 /= np.linalg.norm(out_vector_1, ord=self.norm_ord)
            out_vector_2 /= np.linalg.norm(out_vector_2, ord=self.norm_ord)

        return np.linalg.norm(out_vector_1 - out_vector_2, ord=self.norm_ord)


def load_words_vectors(words_vectors_dir: str) \
        -> tp.Generator[tp.Tuple[str, tp.List[str], tp.List[tp.Any], tp.List[tp.Any]], None, None]:
    for word_path in Path(words_vectors_dir).iterdir():
        with open(word_path) as f:
            word_pairs = json.load(f)
        word = word_path.with_suffix('').name
        pairs_ids = [pair['id'] for pair in word_pairs]
        vectors_1 = [pair['vector_1'] for pair in word_pairs]
        vectors_2 = [pair['vector_2'] for pair in word_pairs]
        yield word, pairs_ids, vectors_1, vectors_2


def run_similarity(words_vectors_dir: str = 'data/words_vectors/', output_path: str = 'data/graded_scores.tsv',
                   normalize: bool = True, norm_ord: int = 1) -> None:
    predictor = VectorsDistPredictor(normalize=normalize, norm_ord=norm_ord)
    words_preds = {}

    for word, _, vectors_1, vectors_2 in tqdm(load_words_vectors(words_vectors_dir),
                                              total=get_files_num(words_vectors_dir)):
        preds = [predictor.predict(vector_1, vector_2) for vector_1, vector_2 in zip(vectors_1, vectors_2)]
        words_preds[word] = np.mean(preds)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as out:
        out.write(f'word\tchange_graded\tCOMPARE\n')
        for word, pred in words_preds.items():
            out.write(f'{word}\t{pred}\t{pred}\n')


if __name__ == '__main__':
    fire.Fire(run_similarity)
