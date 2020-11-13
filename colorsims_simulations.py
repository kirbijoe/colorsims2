import numpy as np
from os import mkdir, path
from progressbar import ProgressBar, Bar, Percentage, ETA
from matplotlib import pyplot as plt
from matplotlib import colors
import time
import math
import colorsims_utils


class LearningAndCommunicationGame():
    '''A learning and communication game simulation played on a hue circle.'''

    def __init__(self, population, color_circle, ksim=3):
        '''Initialize the simulation.
        Assumes population is an AgentPopulation object and color_circle is a DiscreteCircle object.
        Assumes ksim is a positive integer.'''

        self.population = population
        self.color_circle = color_circle
        self.ksim = ksim

        #self.game_history = []

        self.folder = 'demosim'	#name of folder where all simulation files will be saved

    def play_random_games(self, num_games, save_to_disk=False, save_every=100, save_path=''):
        '''Simulates a specified number of games where each game involves two agents
        randomly selected from the population.

        If you want to periodically save population to disk specify save_to_disk=True.
        Also specify how often you want to save in save_every, and the path in save_path. Default
        is current working path.'''

        #Initialize the progress bar
        progress = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=num_games).start()

        if save_to_disk:
            #Make sure the path does not already exist. If it does, a new file name will be created adding '_n' to the end of
            #save_path, where n is a positive integer and is determined by whatever the last instance of save_path was saved as.
            while path.exists(save_path):
                #save_path already exists
                try:
                    old_file_num = int(save_path[save_path.find('_')+1:])
                    new_file_num = old_file_num + 1
                    save_path = save_path[0:save_path.find('_')] + '_' + str(new_file_num)
                except ValueError:
                    save_path = save_path + "_1"

            self.folder = save_path			
            mkdir(save_path)
            self.population.save(path=save_path, suffix='0')

        # Pre-sample the stimuli
        stimuli = self.color_circle.sample([num_games, 2])

        #Run the games
        for i in range(num_games):
            #Randomly select the agents and the stimuli for a single game
            agent_keys = self.population.get_random_pair()

            # Save the game history
            #self.game_history.append([agent_keys, stimuli[i]])

            # Play the game
            self.play_game(agent_keys, stimuli[i])

            #Update the status bar
            progress.update(i + 1)

            #Save to disk
            if save_to_disk and (i % save_every) == 0:
                self.population.save(path=save_path, suffix=str(i))

        # Save the last iteration if it was not already saved
        if save_to_disk and (i % save_every) != 0:
            self.population.save(path=save_path, suffix=str(i))

        #End the status bar
        progress.finish()

    def learning_update(self, agent, stimuli, responses, responses_to_learn):
        agent.punish(stimuli[0], responses[0], response_to_learn=responses_to_learn[0])
        agent.punish(stimuli[1], responses[1], response_to_learn=responses_to_learn[1])

    def failure_update(self, agent, stimuli, responses):
        agent.punish(stimuli[0], responses[0])
        agent.punish(stimuli[1], responses[1])

    def success_update(self, agent, stimuli, responses):
        agent.reinforce(stimuli[0], responses[0])
        agent.reinforce(stimuli[1], responses[1])

    def play_game(self, agents, stimuli):
        '''Play a game between agents using stimuli.
        Assumes agents is a tuple or list of two agent objects and stimuli is a tuple or list of two color chips. '''

        agent_a, agent_b = self.population.get_agent(agents[0]), self.population.get_agent(agents[1])
        #Make sure the stimuli are sorted (required for some comparisons such as computing distance)
        stim1, stim2 = sorted(stimuli)

        #Are the stimuli within ksim distance of each other?
        stim_within_ksim = self.color_circle.distance(stim1, stim2) <= self.ksim

        #Each agent generates a response for each stimulus
        response_a = [ agent_a.response(stim1), agent_a.response(stim2) ]
        response_b = [ agent_b.response(stim1), agent_b.response(stim2) ]

        #Determine if agents had personal successes
        if stim_within_ksim:
            #stimuli were within ksim so agents must use same category for both chips in order to have personal success
            personal_success_a = response_a[0] == response_a[1]
            personal_success_b = response_b[0] == response_b[1]
        else:
            #stimuli were outside of ksim distance so agents must use different categories for each chip in order to have personal success
            personal_success_a = response_a[0] != response_a[1]
            personal_success_b = response_b[0] != response_b[1]

        if personal_success_a and personal_success_b:
            #They both succeeded so choose one to be the student at random by altering their outcome to failure.
            if np.random.randint(2):	#toss a coin to choose an agent
                personal_success_a = False
            else:
                personal_success_b = False

        if personal_success_a:
            # A is the teacher and B is the student
            self.success_update(agent_a, [stim1, stim2], response_a)
            self.learning_update(agent_b, [stim1, stim2], response_b, response_a)

        elif personal_success_b:
            # B is the teacher and A is the student
            self.success_update(agent_b, [stim1, stim2], response_b)
            self.learning_update(agent_a, [stim1, stim2], response_a, response_b)

        else: # both agents failed
            # No teacher in this case
            self.failure_update(agent_a, [stim1, stim2], response_a)
            self.failure_update(agent_b, [stim1, stim2], response_b)
			
			


class LearningAndCommunicationGame_Grid(LearningAndCommunicationGame):
    '''A learning and communication game simulation on a color grid.'''
 
    def __init__(self, population, color_grid, ksim=1, metric="", forced_rounds=200000, epsilon=0.00175):
        '''Initialize the simulation.
        Assumes population is an AgentPopulation object and color_grid is a ColorGrid object.
        Assumes ksim is a positive integer.
        metric can equal {"", "city block", "CIELUV"}.'''
 
        self.population = population
        self.color_grid = color_grid
        self.ksim = ksim
        self.metric_type = metric
        self.num_forced_rounds = forced_rounds
        self.epsilon = epsilon
 
        #self.game_history = []

        self.folder = "demosim"	#name of folder where all simulation files will be saved
		
        conversions = path.abspath("MunsellCIE.txt")		
		
        conv_file = open(conversions, "r")
        self.LAB_table = self.color_grid.LAB_lookup(conv_file)
        self.LUV_table = self.color_grid.LUV_lookup(self.LAB_table)
		
        self.row_convert = {0:'B', 1:'C', 2:'D', 3:'E', 4:'F', 5:'G', 6:'H', 7:'I'}
		
        self.global_agree = 0
        self.within_range = 0
        self.save_every = 0
        self.stop_rounds = 0
 
 
    def play_random_games(self, num_games, save_to_disk=False, save_every=100, save_path=''):		
        '''Simulates a specified number of games where each game involves two agents
        randomly selected from the population.
 
        If you want to periodically save population to disk specify save_to_disk=True.
        Also specify how often you want to save in save_every, and the path in save_path. Default
        is current working path.'''
 
        #Initialize the progress bar
        progress = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=num_games).start()
 
        if save_to_disk:
            # Make sure the path does not already exist
            while path.exists(save_path):
                try:
                    old_file_num = int(save_path[save_path.find('_')+1:])
                    new_file_num = old_file_num + 1
                    save_path = save_path[0:save_path.find('_')] + '_' + str(new_file_num)
                except ValueError:
                    save_path = save_path + "_1"

            self.folder = save_path			
            mkdir(save_path)
            self.population.save(path=save_path, suffix='0')

        #Pre-sample the stimuli
        stimuli = self.color_grid.sample(num_games, 2)		# list of lists (len=2), where each element is a 2-tuple representing a chip

        #Run the games
        for i in range(num_games):
 			
            #Randomly select the agents for a single game
            agent_keys = self.population.get_random_pair()
 
            #Save the game history
            #self.game_history.append([agent_keys, stimuli[i]])	# stimuli[i] = [chip1, chip2] where chip1,2 are 2-tuples
 
            #Play the game
            self.play_game(agent_keys, stimuli[i])
 
            #Update the status bar
            progress.update(i + 1)

            #Save to disk
            if save_to_disk and (i % save_every) == 0:
                self.population.save(path=save_path, suffix=str(i))
 
        # Save the last iteration if it was not already saved
        if save_to_disk and (i % save_every) != 0:
            self.population.save(path=save_path, suffix=str(i))
 
        #End the status bar
        progress.finish()
	

    def play_all_games(self, WCS=False, save_to_disk=True, save_every=10000, save_path=''):
        '''
        Simulates a specified number of games where each game involves two agents
        randomly selected from the population.
 
        If you want to periodically save population to disk specify save_to_disk=True.
        Also specify how often you want to save in save_every, and the path in save_path. Default
        is current working path.

        WCS is a boolean that represents whether the initial population is random or taken from 
        the World Color Survey.
        WCS=False: Random initial population
        WCS=True: World Color Survey language population
		
        Number of games not specified. Simulation will run until reach stable solution.

        This play_games function will plot the population snapshots concurrently as the simulation progresses.
        '''
			
        self.save_every = save_every
 
        if save_to_disk:
            # Make sure the path does not already exist
            while path.exists(save_path):
                try:
                    old_file_num = int(save_path[save_path.find('_')+1:])
                    new_file_num = old_file_num + 1
                    save_path = save_path[0:save_path.find('_')] + '_' + str(new_file_num)
                except ValueError:
                    save_path = save_path + "_1"

            self.folder = save_path			
            mkdir(save_path)
            self.population.save(path=save_path, suffix='0')
 
        write_file = open(path.abspath(path.join(self.folder,"agreement_levels.txt")), 'w')
        chip_agreement = open(path.abspath(path.join(self.folder,"chip_agreement.txt")), 'w')

        V = colorsims_utils.SimulationVisualizer(self.folder)
        if not path.exists(V.image_path):
            mkdir(V.image_path)

        initial_agreement = self._calc_agree()	#Used for calculating language stability at the end of the simulation
        self.global_agree = self._calc_agree()

        write_file.write("{:d}\t{:4f}\n".format(0, self.global_agree))	#Add initial data (game_num=0) to stability measure log (agreement_levels.txt)
        first_line_chipagree = "Round Number"
        for i in range(320):
            first_line_chipagree += "\t"+str(i)
        first_line_chipagree += "\n"
        chip_agreement.write(first_line_chipagree) #Write headers for chip agreement level log (chip_agreement.txt)

        game_num = 1

        print("Starting game simulations...")
        start = time.time()
        while True:
            #Randomly select the agents and the stimuli for a single game
            agent_keys = self.population.get_random_pair()
            stimuli = self.color_grid.sample()		#stimuli = [chip1, chip2] where chip1,2 are 2-tuples
 
            #Play the game
            self.play_game(agent_keys, stimuli[0])
 
            #Save population snapshot to disk
            if save_to_disk and (game_num % save_every) == 0:
                self.population.save(path=save_path, suffix=str(game_num))
                filename = 'pop' + str(game_num) + '.pkl'

                try:
                    population = colorsims_utils.load_population(V.data_path + '/' + filename)
                except:
                    print(V.data_path + '/' + filename)
                    raise

                # The filename for the plot will be in the snapshots dir, using the same filename as the saved
                # population datafile, but with the extension changed to reflect that it's an image file.
                plot_filename = V.image_path + '/' + filename.split('.')[0] + '.png'

                # Create and save the image
                population.plot_population(filename=plot_filename)
 			
            #Check level of agreement
            if game_num % save_every == 0:
                previous_agree = self.global_agree
                change_agree = self._calc_change_agree()
                percent_change = change_agree/(previous_agree + self.global_agree)
                write_file.write("{:d}\t{:4f}\t{:4f}\t{:6f}\n".format(game_num, self._calc_agree(), change_agree, percent_change))
                self.plot_heat_map(game_num, chip_agreement)
                if WCS==False or (WCS==True and game_num >= self.num_forced_rounds):	
                    if self._check_stable(percent_change, game_num):
                        break

            if game_num % 100000 == 0:
                lap = time.time()
                elapsed_time = lap - start
                print("Game number: {} \t | Elapsed time: {:02d}:{:02d}:{:02d}".format(game_num, int(elapsed_time//3600), int((elapsed_time-(elapsed_time//3600)*3600)//60), int(elapsed_time%60))) 

            game_num += 1
 
		# Save the last iteration if it was not already saved
        if save_to_disk and (game_num % save_every) != 0:
            self.population.save(path=save_path, suffix=str(game_num))

        write_file.close()
        chip_agreement.close()

        sys_solution = open(path.abspath(path.join(self.folder,"system_solution.csv")), 'w')
        for i in range(self.population.num_agents):
            for j in range(self.color_grid.num_chips):
                sys_solution.write(str(np.matrix.flatten(self.population.get_agent(i).word_map())[j])+",")
            sys_solution.write("\n")
        sys_solution.close()		
 		
        end = time.time()
        total_time = end - start
        print("Finished running simulations \t  Total run time: {:02d}:{:02d}:{:02d}".format(int(total_time//3600), int((total_time-(total_time//3600)*3600)//60), int(total_time%60))) 


    def play_game(self, agents, stimuli):
        '''Play a game between agents using stimuli.
        Assumes agents is a tuple or list of two agent objects and stimuli is a tuple or list of two color chips.'''
 
        agent_a, agent_b = self.population.get_agent(agents[0]), self.population.get_agent(agents[1])
        #make sure the stimuli are sorted (required for some comparisons such as computing distance)
        stim1, stim2 = sorted(stimuli)
 
        #Are the stimuli within ksim distance of each other?
        if self.metric_type == "city block":
            stim_within_ksim = self.color_grid.city_block_metric(stim1, stim2) <= self.ksim
        elif self.metric_type == "CIELUV":
            Mun_stim1 = (self.row_convert[stim1[0]], str(stim1[1]+1))	#stimulus form: (row, col) 
            Mun_stim2 = (self.row_convert[stim2[0]], str(stim2[1]+1))

            LAB_stim1 = self.LAB_table[Mun_stim1]	#stimulus form: (L,a,b)
            LAB_stim2 = self.LAB_table[Mun_stim2]
			
            LUV_stim1 = self.LUV_table[LAB_stim1]	#stimulus form: (L,u,v)
            LUV_stim2 = self.LUV_table[LAB_stim2]
            stim_within_ksim = self.color_grid.Euclid_distance(LUV_stim1, LUV_stim2) <= self.ksim
        else:	#default metric
            stim_within_ksim = self.color_grid.distance(stim1, stim2) <= self.ksim
 
        #Each agent generates a response for each stimulus
        response_a = [ agent_a.response(stim1), agent_a.response(stim2) ]
        response_b = [ agent_b.response(stim1), agent_b.response(stim2) ]
 
        #Determine if agents had personal successes
        if stim_within_ksim:
            #stimuli were within ksim so agents must use same category for both chips in order to have personal success
            personal_success_a = response_a[0] == response_a[1]
            personal_success_b = response_b[0] == response_b[1]
        else:
            #stimuli were outside of ksim distance so agents must use different categories for each chip in order to have personal success
            personal_success_a = response_a[0] != response_a[1]
            personal_success_b = response_b[0] != response_b[1]
 
        if personal_success_a and personal_success_b:
            #They both succeeded so choose one to be the student at random by altering their outcome to failure.
            if np.random.randint(2): # toss a coin to choose an agent
                personal_success_a = False
            else:
                personal_success_b = False
 
        if personal_success_a:
            # A is the teacher and B is the student
            self.success_update(agent_a, [stim1, stim2], response_a)
            self.learning_update(agent_b, [stim1, stim2], response_b, response_a)
 
        elif personal_success_b:
            # B is the teacher and A is the student
            self.success_update(agent_b, [stim1, stim2], response_b)
            self.learning_update(agent_a, [stim1, stim2], response_a, response_b)
 
        else: # both agents failed
            # No teacher in this case
            self.failure_update(agent_a, [stim1, stim2], response_a)
            self.failure_update(agent_b, [stim1, stim2], response_b)


    def plot_heat_map(self, game_num, file_obj):	#file_name is the name of the chip agreement level log for a WCS simulation
        heat_map = np.ndarray(shape=[8,40])
		
        for chip in range(320):
            word_count = {}
            for category in self.population.vocabulary:
                word_count[category] = 0
				 
            for agent_key in range(self.population.num_agents):
                chip_name = self.population.get_agent(agent_key).word_map()[chip//40, chip%40]
                word_count[chip_name] += 1
	
            heat_map[chip//40, chip%40] = float(max(word_count.values()))/self.population.num_agents
		
        file_line = str(game_num)
        for i in heat_map.flatten():
            file_line += "\t"+str(i)
        file_line += "\n"
        file_obj.write(file_line)

        image_path = path.join(self.folder, self.folder + '_heatmaps')
        if not path.exists(image_path):
            mkdir(image_path)

        ax = None
        fig, ax = plt.subplots()

        ax.matshow(heat_map, cmap="hot")

        ax.set_yticks([])
        ax.set_yticklabels([])
   
        ax.set_xticks([])
        ax.set_xticklabels([])
		
        plt.savefig(path.join(image_path, str(game_num)+".png"))	#file name should be game_num.png
		

    def _calc_agree(self):
        current_agreement = 0
        chips_with_name = 0
        
        pop_naming_sys = np.zeros(shape = [self.population.num_agents, 320])
        for agent_key in range(self.population.num_agents):
            pop_naming_sys[agent_key] = self.population.get_agent(agent_key).word_map().flatten()

        for chip in range(320):
            names = pop_naming_sys[:, chip]
            if np.any(names != -1):    #at least one agent has a name for chip
                chips_with_name += 1
                names = names[names != -1]  #remove all of the "no names"
                max_word_count = max(np.unique(names, return_counts=True)[1])
            else:
                max_word_count = 0
            
            current_agreement += float(max_word_count) / self.population.num_agents
        
        return current_agreement / chips_with_name

    
    def _calc_change_agree(self):
        current_agreement = self._calc_agree()
        change = current_agreement - self.global_agree
        self.global_agree = current_agreement
        
        return change
	
    def _check_stable(self, percent_change, game_num):
        if abs(percent_change) <= self.epsilon:	#the percent change in agreement level is less than or equal to 0.175% from the previous agreement level
            self.within_range += 1			    #within_range counts the number of snapshots that have "stable" solutions
        else:
            self.within_range = 0
            return False
		
        if self.within_range == self.stop_rounds//self.save_every:	#simulation will stop when solutions have been stable for as long as it took
            return True												#to reach a stable solution in the first place
        elif self.within_range == 1:
            self.stop_rounds = game_num								#self.stop_rounds is the number of rounds it took to become "stable"
            return False											#i.e. the number of rounds still needed to be considered a "stable" solution	
        else:
            return False

