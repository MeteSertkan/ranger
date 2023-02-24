import gzip
import io
from typing import List, Union
import ir_measures
from ranger.retrieval.config_location import RetrievalLocationConfig
from ranger.metric_containers import AggregatedMetrics
from ir_measures.measures import Measure, MultiMeasures


def read_trec_run(file):
    if hasattr(file, 'read'):
        for line in file:
            if line.strip():
                line = line.strip().split()
                if len(line) == 6:
                    query_id, iteration, doc_id, rank, score, tag = line
                elif len(line) == 4:
                    query_id, doc_id, rank, score = line
                else:
                    raise ValueError('Invalid TREC run format in line: ' + str(line))
                yield ir_measures.ScoredDoc(query_id=query_id, doc_id=doc_id, score=float(score))
    elif isinstance(file, str):
        if '\n' in file:
            yield from read_trec_run(io.StringIO(file))
        else:
            reader = gzip.open if file.endswith('.gz') else open
            with reader(file, 'rt') as f:
                yield from read_trec_run(f)

def load_and_compute_metrics(experiment:str, ir_measure:Union[Measure, MultiMeasures], config:RetrievalLocationConfig) -> AggregatedMetrics:
    """Load and compute metrics for all datasets in config for the given experiment.

    Parameters
    ----------    
    experiment : str
        Experiment key in config.

    ir_measure : Measure
        The retrieval measure to compute.

    config : RetrievalLocationConfig
        Configuration object.

    Returns
    -------
    metrics: AggregatedMetrics
        Aggregated metrics for all datasets. 
    """

    metrics = AggregatedMetrics()

    for k in config.display_names.keys():

        qrels = ir_measures.read_trec_qrels(config.qrels[k])
        run = read_trec_run(config.available_rankings[experiment][k])

        per_query_metric = []

        for m in ir_measures.iter_calc([ir_measure], qrels, run):
            per_query_metric.append(m.value)

        metrics.add(per_query_metric)
    
    return metrics