from typing import List, Union
import numpy as np
import scipy.stats as stats

class AggregatedMetrics:
    """Aggregated metrics for all datasets for one experiment. """
    
    def __init__(self):
        self._means: List[float] = []
        self._stdevs: List[float] = []
        self._counts: List[float] = []
        self._metrics: List[np.ndarray] = []

    def add(self, metric):
        self._means.append(np.mean(metric))
        self._stdevs.append(np.std(metric))
        self._counts.append(len(metric))
        self._metrics.append(np.array(metric))
        
    def get_means(self) -> np.ndarray:
        return np.array(self._means)

    def get_stdevs(self) -> np.ndarray:
        return np.array(self._stdevs)
    
    def get_counts(self) -> np.ndarray:
        return np.array(self._counts)
    
    def get_metrics(self) -> List[np.ndarray]:
        return self._metrics

class AggregatedPairedMetrics:
    """Aggregated metrics for all datasets for a pair of experiments.
    (e.g., baseline vs. model)
    """
    def __init__(self,
                 treatment: List[np.ndarray],
                 control: List[np.ndarray]):
        self._mean_diffs: List[float] = []
        self._stdevs_diffs: List[float] = []
        self._corrs: List[float] = []
        self._counts: List[float] = []
        self.computePairedMetrics(treatment, control)

    def computePairedMetrics(self,
                             treatment: List[np.ndarray],
                             control: List[np.ndarray]):
        for t,c in zip(treatment,control):
            self._mean_diffs.append(np.mean(t-c))
            self._stdevs_diffs.append(np.std(t-c))
            r,_ = stats.pearsonr(t, c)
            self._corrs.append(r)
            self._counts.append(len(t))

    def get_mean_diffs(self) -> np.ndarray:
        return np.array(self._mean_diffs)

    def get_stdevs_diffs(self) -> np.ndarray:
        return np.array(self._stdevs_diffs)
    
    def get_corrs(self) -> np.ndarray:
        return np.array(self._corrs)
    
    def get_counts(self) -> np.ndarray:
        return np.array(self._counts)