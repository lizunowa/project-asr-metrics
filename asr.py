import jiwer
from typing import List, Mapping, Tuple, Union
import Levenshtein
import jiwer.transforms as tr

""""Базовый препроцессинг"""
_default_transform = tr.Compose(
    [
        tr.RemoveMultipleSpaces(),
        tr.Strip(),
        tr.SentencesToListOfWords(),
        tr.RemoveEmptyStrings(),
    ]
)

"""
Получение показателей :
    H - количество правильно распознанных слов;
    S - количество операций ручной замены;
    D - число удалений слов;
    I - число вставки слов;
через расстояние Левенштейна
"""
def _get_operation_counts(
    source_string: str, destination_string: str
) -> Tuple[int, int, int, int]:

    editops = Levenshtein.editops(source_string, destination_string)

    substitutions = sum(1 if op[0] == "replace" else 0 for op in editops)
    deletions = sum(1 if op[0] == "delete" else 0 for op in editops)
    insertions = sum(1 if op[0] == "insert" else 0 for op in editops)
    hits = len(source_string) - (substitutions + deletions)

    return hits, substitutions, deletions, insertions

"""Подсчет метрик"""

""""Метрики, реализованные ранее в библиотеке"""
#Word Error Rate (WER)
def asr_wer(truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs ) -> float:
    return jiwer.wer(truth, hypothesis, truth_transform, hypothesis_transform)

# Match Error Rate (MER)
def asr_mer(truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs) -> float:
    return jiwer.mer(truth, hypothesis, truth_transform, hypothesis_transform)

# Word Information Preserved (WIP)
def asr_wip(truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs) -> float:
    return jiwer.mer(truth, hypothesis, truth_transform, hypothesis_transform)

# Word Information Lost (WIL)
def asr_wil(truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs) -> float:
    return jiwer.wil(truth, hypothesis, truth_transform, hypothesis_transform)

""""Метрики, реализованные ранее в библиотеке"""

# Word Recognition Rate (WRR)
def asr_wrr(truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs ) -> float:
    return 1 - jiwer.wer(truth, hypothesis, truth_transform, hypothesis_transform)

# Word Correctly Recognized (WCR)
def asr_wсr(truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs ) -> float:
    H, S, D, I = _get_operation_counts(truth, hypothesis)
    wcr = (float(len(hypothesis) - D - S))/float(H + S + D)
    return wcr


""""Создание массива с метриками"""

def all(truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs):
    all_m = []
    all_m.append(asr_wer(truth, hypothesis, truth_transform, hypothesis_transform))
    all_m.append(asr_wrr(truth, hypothesis, truth_transform, hypothesis_transform))
    all_m.append(asr_mer(truth, hypothesis, truth_transform, hypothesis_transform))
    all_m.append(asr_wip(truth, hypothesis, truth_transform, hypothesis_transform))
    all_m.append(asr_wil(truth, hypothesis, truth_transform, hypothesis_transform))
    all_m.append(asr_wсr(truth, hypothesis, truth_transform, hypothesis_transform))

    return all_m


def all_metrics_map(truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs)-> Mapping[str, float]:

    wer = asr_wer(truth, hypothesis, truth_transform, hypothesis_transform)
    wrr = asr_wrr(truth, hypothesis, truth_transform, hypothesis_transform)
    mer = asr_mer(truth, hypothesis, truth_transform, hypothesis_transform)
    wcr = asr_wсr(truth, hypothesis, truth_transform, hypothesis_transform)
    wil = asr_wil(truth, hypothesis, truth_transform, hypothesis_transform)
    wip = asr_wip(truth, hypothesis, truth_transform, hypothesis_transform)

    return {
        "wer": wer,
        "wrr": wrr,
        "mer": mer,
        "wcr": wcr,
        "wil": wil,
        "wip": wip,
    }
