import numpy as np
from progressbar import ProgressBar, Bar, Percentage, ETA
import pickle
from os import listdir, mkdir, path
from os.path import isfile, join, exists
import re
import moviepy.editor as mpy
import pandas as pd



def load_population(filename):
	'''loads a saved AgentPopulation object from disk.'''
	
	f = open(filename, 'rb')
	pop = pickle.load(f)
	f.close()
	return pop

def find_ksim(num_BCT):
    ksim = (-2+(4-(-8)*((320/num_BCT)**2))**(0.5))/4
    return ksim

def convert_rowcol_to_index(chip):
    if chip[0] == 'A' or chip[0] == 'J' or chip[1:] =='0':
        return -1
    else:
        row_convert = {'B':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7}
        row = row_convert[chip[0]]
        col = int(chip[1:]) - 1
        return 40*row + col

def language_dict():
    term_enum = np.array(pd.read_table(path.abspath("dict.txt")))
    lang_dict = dict()
    for lang_num in np.unique(term_enum[:,0])[1:]:
        term_dict = dict()
        for row in term_enum:
            if row[0] == lang_num:
                term_dict[row[3]] = int(row[1])
        lang_dict[int(lang_num)] = term_dict
    return lang_dict

def bct_lookup():
    bct_data = np.array(pd.read_table(path.abspath("all_bcts.txt"), encoding='utf-8'))
    bct_dict = dict()
    for lang_num in np.unique(bct_data[:,0][1:]):
        lang_bcts = []
        for row in bct_data:
            if row[0] == lang_num:
               lang_bcts.append(int(row[1]))
        bct_dict[int(lang_num)] = lang_bcts
    return bct_dict

def convert_index_to_rowcol(chip_index):
    #chip_index can be any number from 1 to 320 

    row = (chip_index-1)//40
    col = (chip_index-1)%40
    row_convert = {0:'B', 1:'C', 2:'D', 3:'E', 4:'F', 5:'G', 6:'H', 7:'I'}
    return row_convert[row] + str(col+1)



class SimulationVisualizer():

	def __init__(self, path_to_simulation_data_files):
		'''Assumes path_to_simulation_data_files is a string specifying a path containing population snapshot files with .pkl extension'''
		
		self.data_path = path_to_simulation_data_files
		self.image_path = self.data_path + '/snapshots'

	
	def plot_population_snapshots(self, extension='png'):
		'''Creates a series of plots for a number of saved population objects that were generated during a simulation.
		Saves the plots as image files in a new subpath specified by image_path.
		Filetype can be specified using the extension parameter. Default is pdf.
		Valid extensions are anything that is handled by your matplotlib installation.
		All files are assumed to be binary pickled population objects.'''

		# Get a list of all the files
		files = [f for f in listdir(self.data_path) if isfile(join(self.data_path,f)) and f.split('.')[1]=='pkl']

		# Check if the image_path directory exists and create if necessary
		if not exists(self.image_path):
			mkdir(self.image_path)

		#Initialize the progress bar
		progress = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(files)).start()

		# Create all of the plots
		for i, filename in enumerate(files):
			# load the population data
			try:
				population = load_population(self.data_path + '/' + filename)
			except:
				print(self.data_path + '/' + filename)
				raise

			# The filename for the plot will be in the snapshots dir, using the same filename as the saved
			# population datafile, but with the extension changed to reflect that it's an image file.
			plot_filename = self.image_path + '/' + filename.split('.')[0] + '.' + extension

			# Create and save the image
			population.plot_population(filename=plot_filename)

			#Update the status bar
			progress.update(i + 1)

		#End the status bar
		progress.finish()

	def movie_from_snapshots(self, filename='evo_movie', image_extension='png', frames_per_second=4):
		'''Creates a movies from the population snapshot images.
		Assumes the images already exist in self.image_path directory and all have extension image_extension.'''

		# Get a list of the image filenames that have the extension specified by image_extension
		image_files = [self.image_path+'/'+f for f in listdir(self.image_path) if isfile(join(self.image_path,f)) and f.split('.')[1]==image_extension]

		# Sort the files in ascending order based on iteration number
		self._sort_nicely(image_files)

		# Create the image sequence
		image_sequence = mpy.ImageSequenceClip(image_files, fps=frames_per_second)

		# Save the movie
		image_sequence.write_videofile(filename+'.mp4', fps=frames_per_second)

		return image_sequence

	def _tryint(self, s):
		'''converts a string to integers or returns the string if string cannot be converted.'''
		
		try:
			return int(s)
		except:
			return s

	def _alphanum_key(self, s):
		'''Turn a string into a list of string and number chunks.
		e.g. "z23a" -> ["z", 23, "a"]'''
		
		return [ self._tryint(c) for c in re.split('([0-9]+)', s) ]

	def _sort_nicely(self, l):
		'''Sort the given list in the way that humans expect.'''
		
		l.sort(key=self._alphanum_key)



class OptimalStrategyTools():

	def __init__(self):
		pass

	def optimal_number_of_categories(self, num_chips, ksim):
		'''Returns the optimal category size for the given parameters.
		Based on Equation 8 from Komarova, N.L., Jameson, K.A. and Narens, L. (2007) 
		Evolutionary models of color categorization based on discrimination.
		Journal of Mathematical Psychology 51(6):359--382.'''

		return num_chips * (2 * ksim * (ksim + 1))**(-1./2)



