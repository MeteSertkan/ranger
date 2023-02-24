import os

from ranger.classification.data_pipeline import load_and_compute_metrics
from ranger.classification.config_location import ClassificationLocationConfig

config = ClassificationLocationConfig("./examples/classification-glue-data/config.yaml")

title = 'BERT vs. DistilBERT'
control_rankings = "bert"  # this is a key in the config.yaml file
treatment_rankings = "distilbert"  # this is also a key in the config.yaml file

control_data = load_and_compute_metrics(control_rankings, config)
treatment_data = load_and_compute_metrics(treatment_rankings, config)


from ranger.metric_containers import AggregatedPairedMetrics, AggregatedMetrics
from ranger.meta_analysis import analyze_effects


effects = AggregatedPairedMetrics(
    treatment=treatment_data.get_metrics(),
    control=control_data.get_metrics(),
    counts=treatment_data.get_counts()
)
effect_size = analyze_effects(list(config.display_names.values()),
                              effects=effects,
                              effect_type="MD")
print(effect_size)


from ranger.forest_plots import forest_plot

plot = forest_plot(
    title=title,
    experiment_names=list(config.display_names.values()),
    label_x_axis="Standardized Mean Difference",
    effect_size=effect_size,
    fig_width=8,
    fig_height=6 # higher height because we have many (9) test sets
)

fig_name = os.path.join(os.getcwd(), "./examples/sample/"+control_rankings+"_vs_"+treatment_rankings)

# save as svg for web display
plot.savefig(fname=fig_name+".svg", dpi=300, bbox_inches='tight')

# save as pdf for latex integration
plot.savefig(fname=fig_name+".pdf", dpi=300, bbox_inches='tight')






