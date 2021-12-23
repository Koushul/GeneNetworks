from pathlib import Path
import json
import pandas as pd
import numpy as np
import scipy.sparse as ssp
import networkx as nx
from scipy.sparse.linalg import inv
from scipy import sparse
from scipy.sparse import SparseEfficiencyWarning
from collections import Counter
import sys
import time
import datetime
import random
import string
import shutil

try:
    import metis
except RuntimeError:
    pass


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', SparseEfficiencyWarning)

sys.path.append("../EM-sCGGM/")
from em_scggm import em_scggm


class NetworkNotCreatedError(Exception):
    pass

class ModelNotTrainedError(Exception):
    pass

class GraphicalModel(object):
    def __init__(self, traits_csv, mutations_csv, rnaseq_csv, name: str = ''):

        assert isinstance(traits_csv, (Path, str))
        assert isinstance(mutations_csv, (Path, str))
        assert isinstance(rnaseq_csv, (Path, str))

        self.name = name
        
        if self.name == '':
            self.name = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(12))
            

        self.traits = pd.read_csv(traits_csv).set_index('Unnamed: 0')
        self.mutations = pd.read_csv(mutations_csv).set_index('Unnamed: 0')
        self.rnaseq = pd.read_csv(rnaseq_csv).set_index('Unnamed: 0')
        self.traits.index.name = None
        self.mutations.index.name = None
        self.rnaseq.index.name = None
 
        self.Z = self.traits.to_numpy()
        self.Y = self.rnaseq.T.to_numpy()
        self.X = self.mutations.to_numpy()

        assert self.X.shape[0]==self.Z.shape[0]
        assert self.Y.shape[0]<=self.X.shape[0]

        self.mutations = self.mutations.T
        self.is_trained = False

    def fit(self, 
            lambdaLambda_z: float, 
            lambdaTheta_yz: float, 
            lambdaLambda_y: float, 
            lambdaTheta_xy: float, 
            max_em_iters = 5, 
            threads = 8,
            verbose = False):
            
        assert isinstance(lambdaLambda_z, float)
        assert isinstance(lambdaTheta_yz, float)
        assert isinstance(lambdaLambda_y, float)
        assert isinstance(lambdaTheta_xy, float)
        assert isinstance(max_em_iters, int)
        assert isinstance(threads, int)
        assert max_em_iters > 0
        assert threads > 0

        self.lambdaLambda_z = lambdaLambda_z
        self.lambdaTheta_yz = lambdaTheta_yz
        self.lambdaLambda_y = lambdaLambda_y
        self.lambdaTheta_xy = lambdaTheta_xy

        self.tic = time.perf_counter()

        (self.Lambda_z, self.Theta_yz, self.Lambda_y, self.Theta_xy, self.Stats) = em_scggm(
            self.Z, 
            self.Y, 
            self.X, 
            lambdaLambda_z, 
            lambdaTheta_yz,
            lambdaLambda_y, 
            lambdaTheta_xy, 
            max_em_iters = max_em_iters, 
            threads = threads,
            verbose = verbose)

        self.toc = time.perf_counter()
        self.is_trained = True

        ## Inference matrices ##
        self.Sigma_z = inv(self.Lambda_z)
        self.Sigma_y = inv(self.Lambda_y)
        self.Sigma_z = self.Sigma_z.reshape(-1, 1)
        
        self.B_xy = -self.Theta_xy * self.Sigma_y # Indirect mutation perturbation effects on gene expression levels
        self.B_yz = -self.Theta_yz * self.Sigma_z # Indirect effects of gene expression levels on traits
        self.B_xz = self.B_xy * self.B_yz         # Mutation effects on traits
        self.B_xz = self.B_xz.reshape(-1, 1)
        self.B_yz = self.B_yz.reshape(-1, 1)
        self.B_yz = sparse.csr_matrix(self.B_yz)
        self.B_xz = sparse.csr_matrix(self.B_xz)
    
        self.Sigma_z_given_x = self.Sigma_z + self.Sigma_z * self.Theta_yz.getH() * self.Sigma_y * self.Theta_yz * self.Sigma_z
        
        ## Joint distribution of p(z,y|x) ##
        self.Lambda_y_given_xz = self.Lambda_y + self.Theta_yz * self.Sigma_z * self.Theta_yz.getH()  # Posterior gene network after seeing phenotype data
        self.Lambda_zy_given_x = np.array([[self.Lambda_z, self.Theta_yz.getH()], [self.Theta_yz, self.Lambda_y_given_xz]])
        self.Theta_zy_given_x = np.array([sparse.csr_matrix(np.zeros((len(self.mutations), len(self.traits)))), self.Theta_xy])
        self.Lambda_y_given_xz = sparse.csr_matrix(self.Lambda_y_given_xz)

    def save(self, path : str):
        """
        Export network weights as sparse Matrices and parameters as json
        """
        if not self.is_trained:
            raise ModelNotTrainedError

        shutil.rmtree(str(path) + '/' + self.name, ignore_errors=True)
        Path(str(path) + '/' + self.name).mkdir()

        save_path = Path(str(path)) / self.name

        ssp.save_npz(save_path / 'Lambda_z.npz', self.Lambda_z)
        ssp.save_npz(save_path / 'Theta_yz.npz', self.Theta_yz)
        ssp.save_npz(save_path / 'Lambda_y.npz', self.Lambda_y)
        ssp.save_npz(save_path / 'Theta_xy.npz', self.Theta_xy)

        t = (self.toc - self.tic)/60.0

        with open(save_path / 'params.json', 'w') as f:
            f.write(json.dumps({'lambdaLambda_z': self.lambdaLambda_z, 
                                'lambdaTheta_yz': self.lambdaTheta_yz, 
                                'lambdaLambda_y': self.lambdaLambda_y, 
                                'lambdaTheta_xy': self.lambdaTheta_xy,
                                'time_elapsed': round(t, 3),
                                'threads': int(sys.argv[1]),
                                'timestamp': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}))                    
        
    def create_network(self, num_modules = None):
        """
        Creates a graph of the gene regulatory network with no isolated nodes.
        If num_modules is specified, genes are grouped into modules.
        """
        self.Network = nx.from_scipy_sparse_matrix(self.Lambda_y_given_xz)
        self.Network = nx.relabel_nodes(self.Network, dict(zip(range(0, len(self.rnaseq)), self.rnaseq.index)))
        self.Network.remove_edges_from(nx.selfloop_edges(self.Network))
        self.Network.remove_nodes_from(set(nx.isolates(self.Network)))

        if num_modules != None:
            _, self.parts = metis.part_graph(self.Network, num_modules, recursive=True)

            self.modules = []
            for i in range(len(Counter(self.parts).keys())):
                self.modules.append(np.where(np.array(self.parts)==i)[0])
            
            print(f"Created Network with {len(self.Network.nodes())} nodes and {num_modules} modules.")
        else:
            print(f"Created Network with {len(self.Network.nodes())} nodes.")
 
    def locate_gene(self, gene : str) -> int: 
        """Find which module a gene is in"""
        assert isinstance(gene, str)
        if not self.parts: raise NetworkNotCreatedError
        return self.parts[list(self.Network.nodes()).index(gene)]
        
    def module_M_on_traits_direct(self, M : int):
        if not self.parts: raise NetworkNotCreatedError
        return sum(abs(self.Theta_yz.tocsr()[np.where(np.array(self.parts) == M)[0], :])).todense()

    def module_M_on_traits_indirect(self, M : int):
        if not self.parts: raise NetworkNotCreatedError
        return sum(abs(self.B_yz.tocsr()[np.where(np.array(self.parts) == M)[0], :])).todense()
    
    def mutation_i_on_traits_mediated_by_M(self, mu : int, M : int):
        if not self.parts: raise NetworkNotCreatedError
        return sum(abs(self.B_xy.tocsr()[mu, np.where(np.array(self.parts) == M)[0]] *
                       self.B_yz.tocsr()[np.where(np.array(self.parts) == M)[0], :])).todense()
    
    # The effects of mutation i on traits
    def mutation_i_on_traits(self, i, trait_ixs):
        return abs(self.B_xz.tocsr()[i,trait_ixs])
    
    def predict_from_expression(self, patient : str = None, custom_expression = None, verbose : int = 0) -> int:
        if not self.is_trained:
            raise ModelNotTrainedError
        assert patient or custom_expression
        if patient:
            exp_matrix = self.rnaseq[patient].values.reshape(1, -1)
        else:
            exp_matrix = custom_expression
        
        if verbose > 0 and patient:
            print(f"Patient {patient} has status {self.traits.loc[patient].values[0]}")

        return 1 if (exp_matrix * self.Theta_yz)[0][0] < 0 else 0
      
    def predict_from_mutations(self, patient : str = None, custom_mutations = None, verbose : int = 0) -> int: 
        if not self.is_trained:
            raise ModelNotTrainedError
        assert patient or custom_mutations
        if patient:
            mu_matrix = self.mutations[patient].values.reshape(1, -1)
        else:
            mu_matrix = custom_mutations
        
        if verbose > 0 and patient:
            print(f"Patient {patient} has status {self.traits.loc[patient].values[0]}")
        
        return 1 if (mu_matrix * self.B_xz)[0][0] > 0 else 0

        
    @property
    def patients(self):
        return self.traits.index.values
    
    @property
    def hpv_connected_genes(self):
        geneloc, _ = np.where(self.Theta_yz.tocsr().todense() != 0)
        return self.rnaseq.iloc[geneloc].index.values.astype(str)
    
    @property
    def shape(self):
        return (self.traits.shape[0], self.traits.shape[1], self.rnaseq.shape[0], self.mutations.shape[0])
    
    def __len__(self):
        return len(self.patients)
        