# Analysis of Taint Ratio on NER

We experiment with the best taint ratio for the training set for a fixed taint ratio on the test set. Each trial is run for 5 times, and the results are in `taint_ratio.csv`.

We provide scripts to run the experiments. You will need to clone `https://github.com/microsoft/unilm/tree/master/layoutlm`, 
and replace `layoutlm/layoutlm/data/funsd.py` and `layoutlm/examples/seq_labeling/run_seq_labeling.py` with the files we provided.

Then you can follow LayoutLM's instruction and run the code. When executing `run_seq_labeling` you can specify 
- train_taint_probability 
- test_taint_probability
- train_taint_seed (random seed)
- test_taint_seed
