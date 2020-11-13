from colorsims_stimuli import DiscreteCircle, ColorGrid
from colorsims_populations import AgentPopulation, SmallWorldNetwork, AgentPopulationOnMoore4Torus
from colorsims_simulations import LearningAndCommunicationGame, LearningAndCommunicationGame_Grid
from colorsims_utils import SimulationVisualizer, find_ksim, bct_lookup
from os import path
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import colors
from colorsims_agents import Agent, RLAgent, RLAgent_Grid
import pandas as pd



def huecircle_sim(n_chips=20, n_agents=9, n_words=5, ksim=3, n_games=10000, save_every=100):
    '''Runs an example simulation using a hue circle color space.'''

    # Create the stimuli: 
    C = DiscreteCircle(n_chips)	#A circle of n_chips discrete color chips
	
    # Create the agent population:
    # First parameter is the "vocabulary" which is a list of "words". It's sufficient
    # to just provide a list of numbers. For example, if there are 5 words: 0,1,2,3,4.
    # Also pass in the stimulus object, C or G.
    # And specify the population size (This value should have a square root that is an
    # integer so the population can be visualized properly).
	
    P = AgentPopulation(list(range(5)), C, n_agents)

    # Initialize the Simulation by passing in the agent population and stimuli
	
    Sim = LearningAndCommunicationGame(P, C, ksim=ksim)

    print ("Running the simulation")
	
    # The first parameter specifies the number of iterations of the game to play.
    # Save the state of the simulation every 'save_every' number of games. Making this 
    # too frequent will cause A LOT of files to be saved to disk and will be very slow.
    # save_path specifies where to save the files.  
    # NOTE: save_path cannot have any underscore characters, '_', because alternate 
    # path names are automatically created if save_path already exists.
    Sim.play_random_games(n_games, save_to_disk=True, save_every=save_every, save_path='demosim')

    # Initialize the visualization tool and tell it which path the simulation data
    # is saved in.
    V = SimulationVisualizer(Sim.folder)

    print ("plotting population snapshots and saving as images")
    V.plot_population_snapshots()
	
    print ("Creating a movie out of the snapshots")
    V.movie_from_snapshots(filename=path.join(Sim.folder, Sim.folder))
	

def random_grid_sim(n_words, n_agents, n_games=None, save_every=10000, k_sim=80):
    '''Runs an example simulation on a population of random agents using the given parameters.
    Color space is assumed to be a color grid (i.e. the 320 chromatic Munsell color chips).
    n_words = number of words in the agents' vocabulary
    n_agents = number of agents in the population
    n_games = number of games to play; if None, then let the simulation end naturally when it converges to a stable solution
    save_every = 
    k_sim = discrimination measure (value depends on metric)'''

    G = ColorGrid()
    P = AgentPopulation(range(n_words), G, n_agents)
    #P = SmallWorldNetwork([0,1,2,3,4], G, 25, 5, 0.5)
    Sim = LearningAndCommunicationGame_Grid(P, G, ksim=k_sim, metric="CIELUV")

    if n_games != None:
        Sim.play_random_games(n_games, save_to_disk=True, save_every=save_every, save_path='demosim')
        V = SimulationVisualizer(Sim.folder)
        V.plot_population_snapshots()
    else:
        Sim.play_all_games(save_to_disk=True, save_every=save_every, save_path='demosim')
        V = SimulationVisualizer(Sim.folder)

    print ("Creating a movie out of the snapshots")
    V.movie_from_snapshots(filename=path.join(Sim.folder, Sim.folder))


lang_bcts = bct_lookup()

def WCS_sim(lang_num, forced_rounds=200000, epsilon=0.00175):
    '''Runs a simulation using the World Color Survey data to model the agent population.
    lang_num is the number of the language as recorded in the World Color Survey.
    forced_rounds is the number of rounds to play before starting to check for stability.
    epsilon is the neighbornood around 0 which determines the stability range (used for determining when a simulation should be stopped).'''
	
    num_BCT = len(lang_bcts[lang_num])
    k_sim = find_ksim(num_BCT)

    directory = path.abspath("WCSParticipantData")
    lang_data = np.array(pd.read_csv(path.join(directory, "ParticipantData", "ParticipantDataLang" + str(lang_num) + ".csv"), header=None))
    num_agents = lang_data.shape[0]
    num_words = max(lang_data.astype(int).flatten())
    words = range(num_words)
	
    G = ColorGrid()
    P = AgentPopulation(words, G, num_agents)
    Sim = LearningAndCommunicationGame_Grid(P, G, ksim=k_sim, metric="CIELUV", forced_rounds=forced_rounds, epsilon=epsilon)
	
    for i in range(num_agents):
        agent = list(P.agents.values())[i]
        agent_file = path.join(directory, "Single Participant Matrices", "Lang"+str(lang_num), "Lang"+str(lang_num)+"Participant"+str(i+1)+".csv")
        matrix = np.array(pd.read_csv(agent_file, header=None)).astype(float)*agent.reinforcement_units
        agent.set_unnormalized_naming_strategy(matrix)
	
    Sim.play_all_games(WCS=True, save_to_disk=True, save_path='Lang'+str(lang_num))

    #Create movie of simulation
    V = SimulationVisualizer(Sim.folder)

    print ("Creating a movie out of the snapshots")
    V.movie_from_snapshots(filename=Sim.folder)

def all_WCS_sims():
    '''Performs a WCS_sim for all the WCS langauges.'''

    numbct_file = open(path.abspath("colorsims NumBCT.csv"), 'r')
    line = numbct_file.readline()
    while line != "":
        line.replace("\n", "")
        print("--------- LANG NUM: " + line.split(",")[0] + " ---------")
        lang_num = int(line.split(",")[0])
        num_BCT = int(line.split(",")[1])
        num_force_rnds = int(line.split(",")[2])
        epsilon = float(line.split(",")[3])
        WCS_sim(lang_num, num_BCT, forced_rounds=num_force_rnds, epsilon=epsilon)
        line = numbct_file.readline()
    numbct_file.close()






	

def WCS_heatmap(lang_num, num_BCT):
    k_sim = find_ksim(num_BCT)

    directory = path.abspath("WCSParticipantData")
    lang_data = np.array(pd.read_csv(path.join(directory, "ParticipantData", "ParticipantDataLang" + str(lang_num) + ".csv"), header=None))
    num_agents = lang_data.shape[0]
    num_words = max(lang_data.astype(int).flatten())
    words = []
    for i in range(num_words):
        words.append(i)

    G = ColorGrid()
    P = SmallWorldNetwork(words, G, num_agents, 5, 0.5)
    Sim = LearningAndCommunicationGame_Grid(P, G, ksim=k_sim, metric="CIELUV")
    Sim.folder = "WCS_heatmaps"
	
    for i in range(num_agents):
        agent = list(P.agents.values())[i]
        agent_file = path.join(directory, "Single Participant Matrices", "Lang"+str(lang_num), "Lang"+str(lang_num)+"Participant"+str(i+1)+".csv")
        matrix = np.array(pd.read_csv(agent_file, header=None)).astype(float)*agent.reinforcement_units
        agent.set_unnormalized_naming_strategy(matrix)
	
    Sim.plot_heat_map(lang_num)


                    

		
    







	
	
