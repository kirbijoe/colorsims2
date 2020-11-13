import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


class Agent():

	def __init__(self, vocabulary):
		'''Initialize an agent with a vocabulary. Assumes vocabulary is a list of integers or strings.'''
		
		self.vocab = vocabulary
		self.num_words = len(vocabulary)



class RLAgent(Agent):
	
	def __init__(self, vocabulary, color_circle, params):
		'''
		Initializes a Reinforcement Learning Agent.
		Assumes vocabulary is a list of integers or strings.
		Assumes color_circle is an instance of DiscreteCircle.

		params is a dictionary of the following optional parameters:

		reinforcement_units: if specified, assumes it is a positive integer. Defaults to 4 * vocab_size.
		The RL mechanism involves moving a subset of units from cell to cell within the
		unnormalized naming strategy matrix.

		reinforcement_delta: The number of reinforcement units used for reinforcement and punishment.
		Default value is (vocab_size * reinforcement_units)/10.

		smoothing_units:  is assumed to be an integer >=0, which is added to the unnormalized strategy
		matrix prior to converting to normalized naming strategy (where each column is a probability
		distribution over words). Default value is 0

		init_method is either "uniform" or "random". Determines if naming_strategy for each stimulus (color chip)
		starts as a uniform or randomized distribution over words. Default value is "random"
		'''

		Agent.__init__(self, vocabulary)
		self.color_circle = color_circle
		
		if 'reinforcement_units' not in params:
			self.reinforcement_units = self.num_words * 4
		else:
			self.reinforcement_units = params['reinforcement_units']

		if 'reinforcement_delta' not in params:
			self.reinforcement_delta = 1.
		else:
			self.reinforcement_delta = params['reinforcement_delta']

		if 'smoothing_units' not in params:
			self.smoothing_units = 0
		else:
			self.smoothing_units = params['smoothing_units']

		#Initialize the unnormalized naming strategy by assigning some number of units to each cell. 
		#Each row in the naming strategy is a color word from the vocabulary and each column is a color chip 
		#on the color circle. Each cell represents the mapping of a color word to a chip on the color circle.
		if 'init_method' not in params:
			init_method = "random"
		else:
			init_method = params['init_method']
		
		if init_method == "random":
			#Assigns a random number of units to each cell
			self.unnormalized_naming_strategy = np.random.multinomial(self.reinforcement_units,
				[1./self.num_words]*self.num_words, size=self.color_circle.chips.shape[0]).T.astype(float)

		elif init_method == "uniform":
			#Assigns a uniform number of units to each cell
			self.unnormalized_naming_strategy = np.ones([self.num_words, self.color_circle.num_chips]) * self.reinforcement_units/self.num_words

	def set_unnormalized_naming_strategy(self, strategy_matrix):
		'''Manually specifies the agent's naming strategy. Assumes strategy_matrix is a two-dimensional numpy array of integers.
		New matrix must be the same shape and have the same number of reinforcement units as the existing naming strategy matrix.'''

		if not strategy_matrix.shape == self.unnormalized_naming_strategy.shape:
			raise Exception('New matrix must be same shape as existing matrix')

		if strategy_matrix.sum() != self.unnormalized_naming_strategy.sum():
			raise Exception('New matrix must have same number of reinforcement units as existing matrix')

		self.unnormalized_naming_strategy = strategy_matrix.copy()
		
	def naming_strategy(self):
		'''Returns the normalized naming strategy using add one smoothing'''
		
		# Add smoothing units to every cell
		smooth_matrix = np.add(self.unnormalized_naming_strategy, self.smoothing_units )

		# Normalize each column by dividing each cell by self.reinforcement_units and return the matrix
		return np.divide(smooth_matrix, smooth_matrix.sum(axis=0))

	def response(self, stimulus, method='match', softmax_param=None):
		'''
		Returns the name of stimulus.
		Assumes stimulus is an integer index into the color circle.

		method:
		
		"match": Probability matching response. Probabilistically returns a name for stimulus
		in proportion to the probability of the name in the stimulus column of the naming strategy.

		"max": Greedy response. Deterministically returns the name with the highest probability for this stimulus.

		"softmax": Samples a response from the naming strategy using a softmax function. Assumes softmax_param is specified.
		This option interpolates between probability matching when softmax_param = 1, and greedy response as the param gets large.
		'''

		if method == "match":
			return np.random.choice( self.vocab, p = self.naming_strategy()[:,stimulus] )

	def reinforce(self, stimulus, response, blocked_responses=[]):
		'''Reinforces the stimulus-response pair.
		Decrements a randomly chosen alternative response to offset the reward and conserve the number
		of reinforcement units for the stimulus.
		The reinforced response and any responses specified in blocked_responses list are not used
		as the decremented response.
		Returns the response that was decremented to offset reinforcement or None if nothing changed.'''

		# Check if the stimulus already has the maximum reinforcement units
		if self.unnormalized_naming_strategy[response, stimulus] < self.reinforcement_units:
			# Doesn't have the max yet so go ahead and reinforce.

			# Figure out the maximum number of reinforcement units that can be added
			delta = self.reinforcement_units - self.unnormalized_naming_strategy[response, stimulus]
			
			# Reinforcement amount will be the equal to reinforcement_delta or however many units need
			# to be added to get to the max for this stimulus -- whichever is lower.
			delta = min(delta, self.reinforcement_delta)

			# Find the responses that can be decremented.
			valid_responses_to_decrement = self.unnormalized_naming_strategy[:, stimulus] >= delta
			# Can't use the response that is being reinforced
			valid_responses_to_decrement[response] = False
			#Can't use any of the blocked responses
			valid_responses_to_decrement[blocked_responses] = False

			if not np.any(valid_responses_to_decrement):
				# No responses can be decremented so can't proceed
				return None

			# Choose one of the valid responses to decrement
			response_to_decrement = np.random.choice(np.arange(self.num_words)[valid_responses_to_decrement])
			# Do the decrement
			self.unnormalized_naming_strategy[response_to_decrement, stimulus] -= delta

			# Do increment for reinforced response
			self.unnormalized_naming_strategy[response, stimulus] += delta

		else:
			# Already at the max reinforcement units for this stimulus.
			return None

		return response_to_decrement

	def punish(self, stimulus, response, blocked_responses=[], response_to_learn=None):
		'''Punishes the stimulus-response pair.
		Increments a randomly chosen alternative response to offset the punishment and conserve the number
		of reinforcement units for the stimulus.
		The reinforced response and any responses specified in blocked_responses list are not used
		as the incremented response.

		Returns the response that was incremented to offset punishment or None if nothing changed.

		If response_to_learn pre-specifies which alternative response should be incremented to offset the punishment.
		This is used when an agent is "learning" a response from another agent, for example.'''

		# Check if the stimulus already has the minimum reinforcement units
		if self.unnormalized_naming_strategy[response, stimulus] > 0:
			# Doesn't have the min yet so go ahead and punish.

			# Figure out the maximum number of reinforcement units that can be subtracted
			delta = self.unnormalized_naming_strategy[response, stimulus]
			
			# Punishment amount will be the equal to reinforcement_delta or however many units need
			# to be subtracted to get to the min for this stimulus -- whichever is lower.
			delta = min(delta, self.reinforcement_delta)

			# Is there a pre-specified response that should be incremented?
			if response_to_learn is not None:
				response_to_increment = response_to_learn

			else:
				#No pre-specified response. Choose a random one.

				# Find the responses that can be incremented.
				valid_responses_to_increment = self.unnormalized_naming_strategy[:, stimulus] <= (self.reinforcement_units - delta)
				# Can't use the response that is being punished
				valid_responses_to_increment[response] = False
				#Can't use any of the blocked responses
				valid_responses_to_increment[blocked_responses] = False
				
				if not np.any(valid_responses_to_increment):
					# No responses can be incremented so can't proceed
					return None

				# Choose one of the valid responses to increment
				response_to_increment = np.random.choice(np.arange(self.num_words)[valid_responses_to_increment])
				
			# Do the increment
			self.unnormalized_naming_strategy[response_to_increment, stimulus] += delta

			# Do decrement for punished response
			self.unnormalized_naming_strategy[response, stimulus] -= delta


		else:
			# Already at the min reinforcement units for this stimulus.
			return None

		return response_to_increment

	def visualize_text_matrix(self):
		'''Returns a text representation of the naming strategy where the word (row)
		with highest probability in each column is a "1" and everything else is "0".'''
		
		m = self.naming_strategy()
		return (m == m.max(axis=0)).astype(int)
	
	def word_map(self):
		'''Returns the index of the of the most probable work for each stimulus (color chip).
		if two or more words are tied for highest probability, the word with the lowest row index is used.'''

		return self.naming_strategy().argmax(axis=0)

	def plot_word_map(self, gray=False, ax=None, filename=None, word_labels=True, stim_labels=True, linewidth=3, color='b', yaxis_label=True, xaxis_label=True):
		'''Plots the agent's naming strategy using a line graph, where the x-axis is the color chips and the y-axis is the color terms.
		Pass filename to save file. Pass optional matplotlib axis object (ax) in order to have plot drawn on the axis.

		gray is a parameter used in RLAgent_Grid. Disregard it when using RLAgent class objects.'''
		
		if ax is None:
			fig, ax = plt.subplots()
		
		#Plot color chips on x-axis and vocabulary on y-axis. 
		ax.plot(self.color_circle.chips, self.word_map(), linewidth=linewidth, color=color)
		ax.set_ylim([-.5, self.vocab[-1]+.5])

		#Show the word labels?
		if word_labels:
			ax.set_yticks(self.vocab)
		else:
			ax.set_yticks(self.vocab)
			ax.set_yticklabels([])

		#Show the color chip labels?
		if stim_labels:
			ax.set_xticks(self.color_circle.chips)
		else:
			ax.set_xticks(self.color_circle.chips)
			ax.set_xticklabels([])

		if yaxis_label:
			ax.set_ylabel('labels')

		if xaxis_label:
			ax.set_xlabel('stimuli')

		#Save to file?
		if filename is not None:
			plt.savefig(filename)



class RLAgent_Grid(RLAgent):

	def __init__(self, vocabulary, color_grid, params):
		'''
		Initializes a Reinforcement Learning Agent.
		Assumes vocabulary is a list of integers or strings.
		Assumes color_grid is an instance of ColorGrid.

		params is a dictionary of the following optional parameters:

		reinforcement_units: if specified, assumes it is a positive integer. Defaults to 4 * vocab_size.
		The RL mechanism involves moving a subset of units from cell to cell within the
		unnormalized naming strategy matrix.

		reinforcement_delta: The number of reinforcement units used for reinforcement and punishment.
		Default value is (vocab_size * reinforcement_units)/10.

		smoothing_units:  is assumed to be an integer >=0, which is added to the unnormalized strategy
		matrix prior to converting to normalized naming strategy (where each column is a probability
		distribution over words). Default value is 0

		init_method is either "uniform" or "random". Determines if naming_strategy for each stimulus (color chip)
		starts as a uniform or randomized distribution over words. Default value is "random"
		'''

		Agent.__init__(self, vocabulary)
		self.color_grid = color_grid
		
		if 'reinforcement_units' not in params:
			self.reinforcement_units = self.num_words * 4
		else:
			self.reinforcement_units = params['reinforcement_units']

		if 'reinforcement_delta' not in params:
			self.reinforcement_delta = 1.
		else:
			self.reinforcement_delta = params['reinforcement_delta']

		if 'smoothing_units' not in params:
			self.smoothing_units = 0
		else:
			self.smoothing_units = params['smoothing_units']

		#Initialize the unnormalized naming strategy by assigning some number of units to each cell. 
		#Each row in the naming strategy is a color word from the vocabulary and each column is a color chip 
		#on the color circle. Each cell represents the mapping of a color word to a chip in the color grid.
		if 'init_method' not in params:
			init_method = "random"
		else:
			init_method = params['init_method']
		
		if init_method == "random":
			#Assigns a uniform number of units to each cell
			self.unnormalized_naming_strategy = np.random.multinomial(self.reinforcement_units,
				[1./self.num_words]*self.num_words, size=self.color_grid.num_chips).T.astype(float)

		elif init_method == "uniform":
			#Assigns a random number of units to each cell
			self.unnormalized_naming_strategy = np.ones([self.num_words, self.color_grid.num_chips]) * self.reinforcement_units/self.num_words

	def response(self, stimulus, method='match', softmax_param=None):
		'''
		Returns the name of stimulus.
		Assumes stimulus is an 2-tuple of integer values, (hue, saturation), that is an index of the color grid.

		method:
		
		"match": Probability matching response. Probabilistically returns a name for stimulus
		in proportion to the probability of the name in the stimulus column of the naming strategy.

		"max": Greedy response. Deterministically returns the name with the highest probability for this stimulus.

		"softmax": Samples a response from the naming strategy using a softmax function. Assumes softmax_param is specified.
		This option interpolates between probability matching when softmax_param = 1, and greedy response as the param gets large.
		'''
		
		if method == "match":
			#stimuli are initialized as 2-tuples where the first value corresponds to the chip's row number (0-7) in the color grid
			#the second value corresponds to the chip's column number (0-39) in the grid.
			stimulus_index = 40*stimulus[0] + stimulus[1]
			return np.random.choice( self.vocab, p = self.naming_strategy()[:,stimulus_index] )

	def reinforce(self, stimulus, response, blocked_responses=[]):
		'''Reinforces the stimulus-response pair. (stimulus is a chip represented by (i,j))
		Decrements a randomly chosen alternative response to offset the reward and conserve the number
		of reinforcement units for the stimulus.
		The reinforced response and any responses specified in blocked_responses list are not used
		as the decremented response.

		Returns the response that was decremented to offset reinforcement or None if nothing changed.'''

		stimulus_index = 40*stimulus[0] + stimulus[1]

		# Check if the stimulus already has the maximum reinforcement units
		if self.unnormalized_naming_strategy[response, stimulus_index] < self.reinforcement_units:
			# Doesn't have the max yet so go ahead and reinforce.

			# Figure out the maximum number of reinforcement units that can be added
			delta = self.reinforcement_units - self.unnormalized_naming_strategy[response, stimulus_index]
			
			# Reinforcement amount will be the equal to reinforcement_delta or however many units need
			# to be added to get to the max for this stimulus -- whichever is lower.
			delta = min(delta, self.reinforcement_delta)

			# Find the responses that can be decremented.
			valid_responses_to_decrement = self.unnormalized_naming_strategy[:, stimulus_index] >= delta
			# Can't use the response that is being reinforced
			valid_responses_to_decrement[response] = False
			#Can't use any of the blocked responses
			valid_responses_to_decrement[blocked_responses] = False

			if not np.any(valid_responses_to_decrement):
				# No responses can be decremented so can't proceed
				return None

			# Choose one of the valid responses to decrement
			response_to_decrement = np.random.choice(np.arange(self.num_words)[valid_responses_to_decrement])
			# Do the decrement
			self.unnormalized_naming_strategy[response_to_decrement, stimulus_index] -= delta

			# Do increment for reinforced response
			self.unnormalized_naming_strategy[response, stimulus_index] += delta

		else:
			# Already at the max reinforcement units for this stimulus.
			return None

		return response_to_decrement

	def punish(self, stimulus, response, blocked_responses=[], response_to_learn=None):
		'''Punishes the stimulus-response pair.  (stimulus is a chip represented by (i,j))
		Increments a randomly chosen alternative response to offset the punishment and conserve the number
		of reinforcement units for the stimulus.
		The reinforced response and any responses specified in blocked_responses list are not used
		as the incremented response.

		Returns the response that was incremented to offset punishment or None if nothing changed.

		If response_to_learn pre-specifies which alternative response should be incremented to offset the punishment.
		This is used when an agent is "learning" a response from another agent, for example. '''

		stimulus_index = 40*stimulus[0] + stimulus[1]

		# Check if the stimulus already has the minimum reinforcement units
		if self.unnormalized_naming_strategy[response, stimulus_index] > 0:
			# Doesn't have the min yet so go ahead and punish.

			# Figure out the maximum number of reinforcement units that can be subtracted
			delta = self.unnormalized_naming_strategy[response, stimulus_index]
			
			# Punishment amount will be the equal to reinforcement_delta or however many units need
			# to be subtracted to get to the min for this stimulus -- whichever is lower.
			delta = min(delta, self.reinforcement_delta)

			# Is there a pre-specified response that should be incremented?
			if response_to_learn is not None:
				response_to_increment = response_to_learn

			else:
				#No pre-specified response. Choose a random one.

				# Find the responses that can be incremented.
				valid_responses_to_increment = self.unnormalized_naming_strategy[:, stimulus_index] <= (self.reinforcement_units - delta)
				# Can't use the response that is being punished
				valid_responses_to_increment[response] = False
				#Can't use any of the blocked responses
				valid_responses_to_increment[blocked_responses] = False
				
				if not np.any(valid_responses_to_increment):
					# No responses can be incremented so can't proceed
					return None

				# Choose one of the valid responses to increment
				response_to_increment = np.random.choice(np.arange(self.num_words)[valid_responses_to_increment])

			# Do the increment
			self.unnormalized_naming_strategy[response_to_increment, stimulus_index] += delta

			# Do decrement for punished response
			self.unnormalized_naming_strategy[response, stimulus_index] -= delta

		else:
			# Already at the min reinforcement units for this stimulus.
			return None

		return response_to_increment

	def word_map(self):
		'''Returns the index of the most probable word for each stimulus (color chip).
		If two or more words are tied for highest probability, the word with the lowest
		row index is used.'''

		word_map = np.full(320, -1, dtype=int)     #This is a 1x320 array filled with -1 (-1 means agent has no name for the chip yet)
		for i in range(320):
			col = self.naming_strategy()[:,i]
			word_map[i] = np.where(col==max(col))[0][0]
		word_map = word_map.reshape([8,40])     #This is an 8x40 matrix with entries being the chosen word for each stimulus.

		return word_map

	def plot_word_map(self, gray=False, ax=None, filename=None, word_labels=True, stim_labels=True, linewidth=3, color='b', yaxis_label=True, xaxis_label=True):
		'''Plots the agent's naming strategy using a heat map, where each of the colors in the map is representative of one of the words 
		in the agent's vocabulary. The map shows what each chip's most probable name is at any instance in the simulation.
		Pass filename to save file.
		Pass optional matplotlib axis object (ax) in order to have plot drawn on the axis.
		gray: prints word map in gray scale if True, in color if False'''
		
		if ax is None:
			fig, ax = plt.subplots()

		#Plot the color map
		if gray:
			ax.matshow(self.word_map(), cmap='gray')
		else:
			#List of 90 colors (list of 30 x3)
			all_colors = ['#8000ff', '#00ff00', '#0040ff', '#cc4400', '#0080ff', '#ff0040', '#ffff00', '#a67c00', '#bf9b30', '#5e3c58', '#bf00ff', '#00ffbf', '#ff0080', 
'#ffdbac', '#000000', '#ff8000', '#d4d4d4', '#555555', '#aafd96', '#00bfff', '#ff00ff', '#ff93ac', '#ffbf00', '#400000', '#204c39', '#997a8d', '#063b79', 
'#757906', '#70330b', '#00ffff', '#8000ff', '#00ff00', '#0040ff', '#cc4400', '#0080ff', '#ff0040', '#ffff00', '#a67c00', '#bf9b30', '#5e3c58', '#bf00ff', 
'#00ffbf', '#ff0080', '#ffdbac', '#000000', '#ff8000', '#d4d4d4', '#555555', '#aafd96', '#00bfff', '#ff00ff', '#ff93ac', '#ffbf00', '#400000', '#204c39', 
'#997a8d', '#063b79', '#757906', '#70330b', '#00ffff', '#8000ff', '#00ff00', '#0040ff', '#cc4400', '#0080ff', '#ff0040', '#ffff00', '#a67c00', '#bf9b30', 
'#5e3c58', '#bf00ff', '#00ffbf', '#ff0080', '#ffdbac', '#000000', '#ff8000', '#d4d4d4', '#555555', '#aafd96', '#00bfff', '#ff00ff', '#ff93ac', '#ffbf00', 
'#400000', '#204c39', '#997a8d', '#063b79', '#757906', '#70330b', '#00ffff']
			cmap = colors.ListedColormap(all_colors)
			bounds = []
			for i in range(len(all_colors)+1):
				bounds.append(i)
			norm = colors.BoundaryNorm(bounds, cmap.N)
			ax.matshow(self.word_map(), cmap=cmap, norm=norm)
		

	 	#Show the word labels?
	 	#Here, the "word labels" corresponds to the number of rows in the color grid.
		if word_labels:
			ax.set_yticks([0,1,2,3,4,5,6,7])
		else:
			ax.set_yticks([0,1,2,3,4,5,6,7])
			ax.set_yticklabels([])

		#Show the color chip labels?
		#Here, the "color chip labels" corresponds to the number of columns in the color grid.
		if stim_labels:
			ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39])
		else:
			ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39])
			ax.set_xticklabels([])
	  
		if yaxis_label:
			ax.set_ylabel('saturation')
	  
		if xaxis_label:
			ax.set_xlabel('hue')					
			
  
		#Save to file?
		if filename is not None:
			plt.savefig(filename)
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		

