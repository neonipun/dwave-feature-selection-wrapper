
import pandas as pd
import dimod
from analytics import *

class QuantumFeatureSelection():

    def __init__(self, op_label="notified", max_features=8, data_file="./cyber_data2.csv", is_notebook=True):
        super().__init__()
        self.title = ""
        self.max_features = max_features
        self.data_file = data_file
        self.op_label = op_label
        self.data = None
        self.mi = {}
        self.features = None
        self.analytics = Analytics()
        self.is_notebook = is_notebook
        self.sorted_mi = None
        self.bqm = None
        self.default_solver = 'DW_2000Q_5'
        self.token = None  #TODO  __token setToken() 
        self.endpoint="https://cloud.dwavesys.com/sapi"
        self.pre_process()
    
    def pre_process(self):
        self.data = pd.read_csv(self.data_file)
        self.features = list(set(self.data.columns).difference((self.op_label,)))
        for feature in self.features:
            self.mi[feature] = self.analytics.mutual_information(self.analytics.prob(self.data[[self.op_label, feature]].values), 1)
        if self.is_notebook:
            from helpers.plots import plot_mi
            print("Printing plot for all features")
            plot_mi(self.mi)
        
        self.sorted_mi = sorted(self.mi.items(), key=lambda pair: pair[1], reverse=True)
        self.data = self.data[[column[0] for column in self.sorted_mi[0:self.max_features]] + [self.op_label]]
        self.features = list(set(self.data.columns).difference((self.op_label,)))

        print("{} no of features selected: {}".format(self.max_features, self.features))
        self.setup_bqm_model()
    
    def setup_bqm_model(self):
        self.bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
        h = {}  #for ising stuff
        temp = {}
        i = 0
        for feature in self.features:
    
            mi = self.analytics.mutual_information(self.analytics.prob(self.data[[self.op_label, feature]].values), 1)
            self.bqm.add_variable(feature, -mi)
            h[i] = -1
            i += 1
            temp[feature] = i

        hh = {}
        import itertools
        for f0, f1 in itertools.combinations(self.features, 2):
            cmi_01 = self.analytics.conditional_mutual_information(self.analytics.prob(self.data[[self.op_label,  f0, f1]].values), 1, 2)
            cmi_10 = self.analytics.conditional_mutual_information(self.analytics.prob(self.data[[self.op_label, f1, f0]].values), 1, 2)
            self.bqm.add_interaction(f0, f1, -cmi_01)
            self.bqm.add_interaction(f1, f0, -cmi_10)

            hh[(temp[f0], temp[f1])] =  -cmi_01     ##for ising purpose
            hh[(temp[f1], temp[f0])] =  -cmi_10
        
        self.bqm.normalize() 
        if self.is_notebook:
            print("plotting BQM model graph")
            from helpers.draw import plot_bqm
            plot_bqm(self.bqm)
        
        print("----- Pre-processing done -----")
            


    def tabu_sampler(self):
        print("\nTabu Sampler....")
        from tabu import TabuSampler
        qpu = TabuSampler()
        self.title = "Tabu Sampler"

        selected_features = np.zeros((len(self.features), len(self.features)))
        for k in range(1, len(self.features) + 1):
            flag = False
            print("Submitting for k={}".format(k))
            kbqm = dimod.generators.combinations(self.features, k, strength=25)
            kbqm.update(self.bqm)
            kbqm.normalize()
        
            while not flag:
                result = qpu.sample(kbqm, num_reads=10)
                # result = qpu.sample_ising(kbqm.to_ising()[0], kbqm.to_ising()[1], num_reads=10)
                best = result.first.sample
            
                if list(result.first.sample.values()).count(1) == k:
                    flag = True
        
            for fi, f in enumerate(self.features):
                selected_features[k-1, fi] = best[f]
        if self.is_notebook:
            from helpers.draw import plot_feature_selection 
            from helpers.plots import plot_solutions
            plot_feature_selection(self.features, selected_features, self.title)

    
    def simulated_annealing(self):
        print("\nSimulated Annealing....")
        import neal
        qpu = neal.SimulatedAnnealingSampler()
        self.title = "Simmulated Annealer"

        selected_features = np.zeros((len(self.features), len(self.features)))
        for k in range(1, len(self.features) + 1):
            flag = False
            print("Submitting for k={}".format(k))
            kbqm = dimod.generators.combinations(self.features, k, strength=25)
            kbqm.update(self.bqm)
            kbqm.normalize()
        
            while not flag:
                result = qpu.sample(kbqm, num_reads=10)
                # result = qpu.sample_ising(kbqm.to_ising()[0], kbqm.to_ising()[1], num_reads=10)
                best = result.first.sample
            
                if list(result.first.sample.values()).count(1) == k:
                    flag = True
        
            for fi, f in enumerate(self.features):
                selected_features[k-1, fi] = best[f]
        if self.is_notebook:
            from helpers.draw import plot_feature_selection 
            from helpers.plots import plot_solutions
            plot_feature_selection(self.features, selected_features, self.title)
    
    def dwave_physical_DW_2000Q_5(self):
        print("\nD-wave quantum annealer....")
        from dwave.system import DWaveSampler, FixedEmbeddingComposite
        from dwave.embedding.chimera import find_clique_embedding

        qpu = DWaveSampler(token=self.token, endpoint=self.endpoint, solver=dict(name='DW_2000Q_5'),auto_scale=False)
        self.title = "D-Wave Quantum Annealer"

        embedding = find_clique_embedding(self.bqm.variables,
                                        16, 16, 4,  # size of the chimera lattice
                                        target_edges=qpu.edgelist)

        qpu_sampler = FixedEmbeddingComposite(qpu, embedding)

        print("Maximum chain length for minor embedding is {}.".format(max(len(x) for x in embedding.values())))

        from hybrid.reference.kerberos import KerberosSampler
        kerberos_sampler = KerberosSampler() 

        selected_features = np.zeros((len(self.features), len(self.features)))
        for k in range(1, len(self.features) + 1):
            print("Submitting for k={}".format(k))
            kbqm = dimod.generators.combinations(self.features, k, strength=6)
            kbqm.update(self.bqm)
            kbqm.normalize()
            
            best = kerberos_sampler.sample(kbqm, qpu_sampler=qpu_sampler, num_reads=10, max_iter=1).first.sample
            
            for fi, f in enumerate(self.features):
                selected_features[k-1, fi] = best[f]
        if self.is_notebook:
            from helpers.draw import plot_feature_selection 
            plot_feature_selection(self.features, selected_features)


if __name__ == "__main__":
    hw4 = QuantumFeatureSelection(is_notebook=False)
    hw4.tabu_sampler()
    hw4.simulated_annealing()
    hw4.dwave_physical_DW_2000Q_5()
    

