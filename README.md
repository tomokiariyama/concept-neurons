# concept neurons
- This repository was created by modifying the codes of https://github.com/EleutherAI/knowledge-neurons.
- This repository is for https://www.anlp.jp/proceedings/annual_meeting/2022/pdf_dir/C4-3.pdf

## How to Reproduce the Figures
- At first, please `git clone` this repository.
- Next, switch to the virtual environment you plan to use while you are in the directory you just `git cloned`(=`concept-neurons`).
  - Note that the required modules will be installed in that virtual environment.
  - We have confirmed the operation with `python 3.8`.
- Next, execute the following command.
  - `chmod +x setup.sh evaluate_ConceptNet.sh make_graphs.sh reproduce_figures.sh`
- Finally, please run `./reproduce_figures.sh`.
- After that, you can find the same figures in `concept_neurons/work/figure/paper/ConceptNet/subject/*` as those in the paper.
  - figure 3: "all_suppressed_graph.png"
  - figure 4: "all_enhanced_graph.png"
  - figure 5: "all_suppressed_overlapping_histogram.png"
  - figure 6: "all_enhanced_overlapping_histogram.png"
  - figure 7: "relevant_suppressed_overlapping_histogram.png"
  - figure 8: "relevant_enhanced_overlapping_histogram.png"


## Scripts
```yaml
setup.sh: Download LAMA dataset and install the required modules to your virtual environment.
evaluate_ConceptNet.sh: Conduct experiments, obtain results.
  - evaluate.py: Experiment codes
make_graphs.sh: Generate graphs.
  - make_graphs.py: Make results into a graph.
```
