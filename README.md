# Profile OmniFold for Top Quark Mass Measurement

This implementation uses OmniFold to unfold the kinematic information of subjets found in large radius top quark jet candidates. To install the required packages do:


## Installation

```bash
pip install requirements.txt
```

Inside a new virtual environment.

## Training

To run the unfolding scripts do:

```bash
cd scripts
python train.py [--verbose] --file_path path/to/root/files
```

Currently, `data' is represented by one of the MC files without weights, while the MC used for the unfolding corresponds to the same simulation with MC weights. Additional options for the unfolding setup are saved in the ```config_omnifold.json``` file, such as which files to use for data and MC.

With the verbose flag, some plots are saved during each iteration of OmniFold. In particular, we can look at how variables change during step 1, where only reconstructed events are used. Those are saved by default in the folder ```plots```

## Evaluating

We can load the trained models and make plots of the unfolded variables using the command:

```bash
python plot.py
```

At the moment, the code only shows the unfolded response of the leading jet pT, but all inputs used during training (4-vector of the 3 subjets) are unfolded simultaneously, so you can implement the same plotting routine for each of them to verify the response.
