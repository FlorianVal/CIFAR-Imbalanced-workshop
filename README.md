# CIFAR workshop on imablanced datasets

This repo contains the code to begin a workshop on CIFAR dataset with imbalanced classes.

To run it :

```bash
python3 generate.py
```

This script will download CIFAR dataset and remove some data to make it imbalanced, by default class 3 is selected to be 1/10th of other classes.

The jupyter notebook check.ipynb is a simple example of how to check and can be used as a boilerplate to start experiments.