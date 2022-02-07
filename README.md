# concept neurons
Now we are preparing this repository. Please wait for a while.

## How to Reproduce Figures
- At first, please `git clone` this repository.
- Next, switch to the virtual environment you'll use when you are in the directory you just `git cloned`(=`concept_neurons`).
- Finally, please execute `zsh reproduce_figures.sh`.
- After that, you can find the same figures which are in the article in `concept_neurons/work/figure/article/ConceptNet/subject/*`.
  - figure 3: "all_suppressed_graph.png"
  - figure 4: "all_enhanced_graph.png"
  - figure 5: "all_suppressed_overlapping_histogram.png"
  - figure 6: "all_enhanced_overlapping_histogram.png"
  - figure 7: "relevant_suppressed_overlapping_histogram.png"
  - figure 8: "relevant_enhanced_overlapping_histogram.png"


## scripts
```yaml
set_dataset_path.sh: Set the directory which LAMA dataset is downloaded. (please change the directory if you needed)
Setup.sh: Download LAMA dataset and install required modules to your virtual environment.
ConceptNet.sh: Conduct experiments, then obtain results.
  - evaluate.py: Experiment codes
make_graphs.sh: Generate graphs.
  - make_graphs.py: Make results into a graph.
```
