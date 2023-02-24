from ranger.classification.config_location import ClassificationLocationConfig
from ranger.metric_containers import AggregatedMetrics

def read_eval_file(file):
    with open(file) as f:
        return list(map(float, f))

def load_and_compute_metrics(experiment: str,
                             config: ClassificationLocationConfig) -> AggregatedMetrics:
    """Load and compute metrics for all datasets in config for the given experiment.

    Parameters
    ----------
    experiment : str
        Experiment key in config.

    config : ClassificationLocationConfig
        Configuration object.

    Returns
    -------
    metrics: AggregatedMetrics
        Aggregated metrics for all datasets.
    """

    metrics = AggregatedMetrics()

    for k in config.display_names.keys():
        metric = read_eval_file(config.eval_metrics[experiment][k])
        metrics.add(metric)
 
    return metrics
