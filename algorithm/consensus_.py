from disropt.algorithms.consensus import Consensus
from disropt.agents.agent import Agent
import numpy as np
import Utils.Utils as uts
class Consensus_(Consensus):

    """Add FastMix to Consensus.

    """

    def __init__(self, agent: Agent, initial_condition: np.ndarray, FastMix: bool = True, enable_log: bool = False):

        super(Consensus, self).__init__(agent, enable_log)

        self.x0 = initial_condition

        self.x = initial_condition

        self.shape = self.x.shape

        self.x_neigh = {}

        self.last_x = initial_condition

        self.is_FastMix = FastMix



    def _update_local_solution_2(self, x: np.ndarray, **kwargs):
        """update the local solution

        Args:
            x: new value

        Raises:
            TypeError: Input must be a numpy.ndarray
            ValueError: Incompatible shapes
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy.ndarray")
        if x.shape != self.x0.shape:
            raise ValueError("Incompatible shapes")
        self.last_x = self.x
        self.x = x

    def iterate_run_(self, **kwargs):

        """Run a single iterate of the algorithm
        """

        data = self.agent.neighbors_exchange(self.x)
        self.last_x = self.x

        for neigh in data:
            self.x_neigh[neigh] = data[neigh]
        x_avg = (1 + self.eta) * self.agent.in_weights[self.agent.id] * self.x - self.eta * self.last_x

        for i in self.agent.in_neighbors:
            x_avg += (1 + self.eta) * self.agent.in_weights[i] * self.x_neigh[i]

        self._update_local_solution_2(x_avg, **kwargs)

    def run_(self, iterations: int = 100, verbose: bool = False, **kwargs):

        """Run the algorithm for a given number of iterations

        Args:
            iterations: Number of iterations. Defaults to 100.
            verbose: If True print some information during the evolution of the algorithm. Defaults to False.
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an int")
        if self.enable_log:
            dims = [iterations]
            for dim in self.x.shape:
                dims.append(dim)
            self.sequence = np.zeros(dims)

        for k in range(iterations):
            self.iterate_run_(**kwargs)

            if self.enable_log:
                self.sequence[k] = self.x

            if verbose:
                if self.agent.id == 0:
                    print('Iteration {}'.format(k), end="\r")

        if self.enable_log:
            return self.sequence
    def get_eta(self, W:np.ndarray):

        if self.is_FastMix == True:
            lamb = uts.find_second_largest(np.linalg.eigvals(W))
            self.eta = (1 - np.sqrt(1 - lamb ** 2)) / (1 + np.sqrt(1 - lamb ** 2))
        else:
            self.eta = 0
