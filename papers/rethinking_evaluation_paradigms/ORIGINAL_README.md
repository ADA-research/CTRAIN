# Rethinking Evaluation Paradigms in IBP-based Certified Training
This is the code appendix to the submission "Rethinking Evaluation Paradigms in IBP-based Certified Training".
It contains all relevant code to rerun all experiments and analyses reported in the paper.

## Setup
First, make sure to have Python installed. All experiments were performed using Python 3.11.5 .
Then, create a virtual environment
```
python3 -m venv ./venv
```
And install all needed requirements.
```
source ./venv/bin/activate
pip3 install -r requirements.txt
```

## Reproduction of Experiments
Notice, that all our experiments were conducted on a cluster that runs SLURM.
Therefore, we use the `submitit` library to schedule SLURM jobs for every experiment. 
However, the present code can easiliy be adapted to run without SLURM.

### HPO
The code to submit all experiments can be found in `experiments/run_hpo.py`.
Notice, that the configuration spaces and the HPO procedure is implemented within our fork of the CTRAIN library.
The configuration spaces can be found in `./MOCTRAIN/CTRAIN/model_wrappers/configs.py` and the optimisation is implemented in
the `hpo` function in `./MOCTRAIN/CTRAIN/model_wrappers/model_wrapper.py`.

### Pareto Front 
As described in the paper, we conduct the HPO across three random seeds and combine the resulting Pareto fronts. 
Please move the resulting optuna_studies into the folder `./results/hpo/optuna_results` and the resulting network checkpoints into the corresponding folder `./results/hpo/{dataset}_{architecture}_{cert_train_method}{epsilon}`. 
Then run `experiments/calculate_fronts.py` to conduct the subselection procedure as described in the paper. Further, this program creates plots of the Pareto fronts discovered by our method. 
The corresponding files that are included in the repository are the results yielded by our runs. 
We postpone the publication of model checkpoints to after acceptance since the files have a comibined size of about 40GB.

### Complete Verification
Then run the complete verification procedure, please run `./experiments/submit_complete_verification.py`. Please notice that the code for the complete verification per network is in `./MOCTRAIN/CTRAIN/eval/eval.py`.
After that, the folder `results/verification` will hold the results yielded by complete verification. It currently holds the results obtained by our experiments. 
To evaluate the networks regarding clean accuracy, run `./eval/eval_nat_acc.py` which will populate `./results/clean_classification`.

### Evaluation
To combine the results regarding certified and natural accuracy, please run `eval/combine_results.py`. 
For the experiments in the Appendix across different architectures, we used a timeout of `300`s and did only consider the first `1000` test samples.
To get the combined results for this case, please adjust `TEST_SAMPLES` and `ARTIFICIAL_TIMEOUT` in `combine_results.py`.
There will then exist results files `./results/verification/summary_results.json` and `./results/verification/summary_results_timeout300_testsamples1000.json`.

To create the parallel coordinates plots run the `./eval/parallel_coordinates.py` script.
This script requires `deepcave` which only supports Python 3.10.
Therefore, we advise to create a separate venv with Python 3.10 and run `pip3 install deepcave==1.3.4 optuna kaleido==0.2.1`. 
The resulting plots can be found in `./eval/importance_analysis`


To obtain results plots and tables, please run `./eval/plot.py` and `./eval/tables.py`. The generated files can be found in `./eval/plots` and `./eval/tables` respectively.
Finally, the analysis of the resulting Pareto fronts, i.e., investigating how each method or network contributes to the combined Pareto fronts can be done by running `./eval/front_analysis.py` and investigating the terminal output.

To investigate the convergence of the Pareto fronts during HPO, run:
```
python ./eval/front_status.py
```
This will generate hypervolume analysis tables in `eval/fronts_development`.

To generate the motivation plot used in the paper introduction (comparing MTL-IBP and SABR with literature), run:
```
python ./eval/plot_motivation.py
```
The plot will be saved to `eval/plots/motivation_sabr_cifar10.pdf`.

To analyze verification running times and generate tables for different timeouts, run:
```
python ./eval/verification_times.py
```
The results will be stored in `eval/tables_verification_times`.

To check the correlation between the incomplete verification (HPO objectives) and the complete verification results, run:
```
python ./eval/verify_correlation.py
```
This generates a correlation table in `eval/tables/verification_correlation.tex`.
