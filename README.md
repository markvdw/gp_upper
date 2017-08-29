# gp_upper
`gp_upper` contains some extra functionality for computing upper bounds to Gaussian process regression marginal
likelihoods, that isn't included in [GPflow](https://github.com/GPflow/GPflow). Currently, this is minimisation of the
upper bound, to tighten it. Usage can be deduced from the notebook, which also discusses how the upper bound can be used
to diagnose over-estimation the marginal likelihood by FITC.

### Installing
The package is pretty tiny, but I added `setup.py` just in case. Install using `python setup.py develop`.

### Testing
A unit test is included for easy verification of correct functioning.

`nosetests testing --nologcapture --with-coverage --cover-package=gp_upper --cover-erase`

### Todo
- Notebook which shows the effect of optimising Z after training with FITC and VFE models.
- Add option for using different bounds on maximum eigenvalue. 

### Notes & thoughts
This project was made to add some extra functionality not absorbed into the GPflow core. While I'll try to keep it
up-to-date, I'm not giving it the same guarantees as GPflow. If something is wrong, or something breaks due to an update
in TensorFlow/GPflow/whatever, feel free to raise an issue or submit a PR.
