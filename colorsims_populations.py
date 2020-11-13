import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import pickle
from colorsims_agents import RLAgent, RLAgent_Grid
from colorsims_stimuli import DiscreteCircle


class AgentPopulation():

	def __init__(self, vocabulary, hue_space, num_agents=9, agent_init_params={}):
		'''
		Initializes a population of agents.
		Assumes vocabulary is a list of the vocabulary words (typically a list of integers).
		Assumes the hue space is either a color_circle or color_grid, where color_circle is a DiscreteCircle object and
		color_grid is a ColorGrid object.
		Assumes num_agents has a square root that is an integer (in order to create a grid of agents with equal height and width).

		agent_init_params is a dictionary of the following optional parameters:

		reinforcement_units: if specified, assumes it is a positive integer. Defaults to 10.
		The RL mechanism involves moving a subset of units from cell to cell within the
		unnormalized naming strategy matrix.

		reinforcement_delta: The number of reinforcement units used for reinforcement and punishment. Default value is 1

		smoothing_units:  is assumed to be an integer >=0, which is added to the unnormalized strategy
		matrix prior to converting to normalized naming strategy (where each column is a probability
		distribution over words). Default value is 1

		init_method is either "uniform" or "random". Determines if naming_strategy for each stimulus (color chip)
		starts as a uniform or randomized distribution over words. Default value is "random"
		'''

		self.vocabulary = vocabulary
		if isinstance(hue_space, DiscreteCircle):
			self.hue_space_type = "color_circle"
			self.color_circle = hue_space
			self.color_grid = None
		else:  #if isinstance(hue_space, ColorGrid)
			self.hue_space_type = "color_grid"
			self.color_grid = hue_space 
			self.color_circle = None
		
		self.num_agents = num_agents
		self.init_network()
		self.agents = {}
		#Populate network with appropriate agent types
		for node in self.network.nodes():
			if self.hue_space_type == "color_circle":
				self.agents[node] = RLAgent(vocabulary, hue_space, params=agent_init_params)
			else:  #if self.hue_space_type == "color_grid"
				self.agents[node] = RLAgent_Grid(vocabulary, hue_space, params=agent_init_params)

	def init_network(self):
		'''Initializes a complete network.
		Override this method in order to init different network types.'''
		
		self.network = nx.complete_graph(self.num_agents)

	def get_agent(self, agent_key):
		'''Returns the agent object specified by the agent_key.'''
		
		return self.agents[agent_key]

	def neighbors(self, agent_key):
		'''Returns the neighbors of the agent specified by agent_key.'''
		
		return self.network.neighbors(agent_key)

	def random_agent(self):
		'''Returns the agent_key of a random agent from the population.'''
		
		return self.network.nodes()[ np.random.choice( np.arange( len(self.network.nodes()) ) ) ]

	def random_neighbor(self, agent_key):
		'''Returns a random neighbor of the agent specified by agent_key'''
		
		# return np.random.choice(self.network.neighbors(agent))
		return self.network.neighbors(agent_key)[ np.random.choice( np.arange( len(self.network.neighbors(agent_key)) ) ) ]

	def get_random_pair(self):
		'''Returns the agent objects for two randomly selected agents who are neighbors of each other'''
		
		agent_one_key = self.random_agent()
		agent_two_key = self.random_neighbor(agent_one_key)

		# return self.get_agent(agent_one_key), self.get_agent(agent_two_key)
		return agent_one_key, agent_two_key

	def random_generation(self):
		'''Pre-generate all of the random agent pairs for a complete generation.'''
		
		pass

	def plot_population(self, gray=False, filename=None):
		'''Plots the population on a grid. The grid is comprised of the visual representation of each agent's word map (line graph 
		for DiscreteCircle and heat map for ColorGrid).
		Override this if you need something different.
		gray: prints word map in gray scale if True, in color if False'''
		
		grid_size = int( np.sqrt(self.num_agents) )
		fig, axarr = plt.subplots(grid_size, grid_size, sharex='col', sharey='row')

		i = 0
		for r in range(grid_size):
			for c in range(grid_size):
				agent = list(self.agents.keys())[i]
				self.get_agent(agent).plot_word_map(gray=gray, ax=axarr[r,c], word_labels=False, stim_labels=False, yaxis_label=False, xaxis_label=False)
				i += 1

		plt.tight_layout()

		if filename is not None:
			plt.savefig(filename)
			plt.close()

	def save(self, path=None, suffix=None):
		'''Saves the population as a binary pickled file.
		Default is in current working directory unless path is specified.
		If suffix is specified it is a string that is appended to the end of the filename.'''

		filename = 'pop'
		
		if path is not None:
			filename = path + '/' + filename

		if suffix is not None:
			filename += suffix

		filename += '.pkl'

		f = open(filename, 'wb')
		pickle.dump(self,f)
		f.close()



class AgentPopulationOnMoore4Torus(AgentPopulation):
	'''Creates a simulation object using a Moore-4 Torus rather than a complete network.'''

	def init_network(self):
		grid_size = int(np.sqrt(self.num_agents))

		# Create a grid where each agent is connected to the four neighbors above, below, left and right.
		self.network = nx.grid_2d_graph(grid_size,grid_size, periodic=True)

		# Add the diagonal neighbors
		for agent_key in self.network.nodes():

			# agent_key = np.array(agent_key)

			# Calculate the adjacent rows
			if agent_key[0] == 0:
				row_above = grid_size - 1
			else:
				row_above = agent_key[0] - 1

			if agent_key[0] == grid_size - 1:
				row_below = 0
			else:
				row_below = agent_key[0] + 1

			# Calculate the adjacent columns
			if agent_key[1] == 0:
				column_left = grid_size - 1
			else:
				column_left = agent_key[1] - 1

			if agent_key[1] == grid_size - 1:
				column_right = 0
			else:
				column_right = agent_key[1] + 1

			# Add edges to the diagonal neighbors
			self.network.add_edge( agent_key, (row_above, column_left) )
			self.network.add_edge( agent_key, (row_above, column_right) )
			self.network.add_edge( agent_key, (row_below, column_left) )
			self.network.add_edge( agent_key, (row_below, column_right) )




class SmallWorldNetwork(AgentPopulation):
	'''Creates a Watts-Strogatz model graph with N nodes, N*K edges, mean node degree 2*K, 
	and rewiring probability beta.
	
	beta = 0 is a ring lattice, and beta = 1 is a random graph.
	'''
	
	def __init__(self, vocabulary, hue_space, N, K, beta, agent_init_params={}):
		self.num_agents = N
		self.K = K
		self.beta = beta
# 		self.num_edges = N*K
# 		self.node_degree = 2*K

		self.vocabulary = vocabulary
		if isinstance(hue_space, DiscreteCircle):
			self.hue_space_type = "color_circle"
			self.color_circle = hue_space
			self.color_grid = None
		else:  #if isinstance(hue_space, ColorGrid)
			self.hue_space_type = "color_grid"
			self.color_grid = hue_space 
			self.color_circle = None
		
		self.init_network()
		self.agents = {}
		for node in self.network.nodes():
			if self.hue_space_type == "color_circle":
				self.agents[node] = RLAgent(vocabulary, hue_space, params=agent_init_params)
			else:  #if self.hue_space_type == "color_grid"
				self.agents[node] = RLAgent_Grid(vocabulary, hue_space, params=agent_init_params)
		
	def init_network(self):
		self.network = nx.watts_strogatz_graph(self.num_agents, self.K, self.beta)
		
