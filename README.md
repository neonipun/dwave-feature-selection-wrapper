# dwave-feature-selection-wrapper
A Quantum Computer course project with @neonipun and @SwapnilBhosale
A Python Wrapper code around the Dwave Feature Selection tutorial from their learning resources.
**Example Data Set used:** [IEEE BigData 2019 Cup: Suspicious Network Event Recognition](https://knowledgepit.ml/suspicious-network-event-recognition/)

## Usage
The Tabu Sampler and Simmulater Annealer can also be run.
```
qfs = QuantumFeatureSelection(is_notebook=False)
qfs.tabu_sampler()
qfs.simulated_annealing()
#qfs.token = "TOKEN"
qfs.dwave_physical_DW_2000Q_5()
```
