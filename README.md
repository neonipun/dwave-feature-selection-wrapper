# dwave-feature-selection-wrapper
A Python Wrapper code around the Dwave Feature Selection tutorial from their learning resources.

## Usage
The Tabu Sampler and Simmulater Annealer can also be run.
```
qfs = QuantumFeatureSelection(is_notebook=False)
qfs.tabu_sampler()
qfs.simulated_annealing()
#qfs.token = "TOKEN"
qfs.dwave_physical_DW_2000Q_5()
```