from Session import Session
from general import gen_pcap_filenames, gen_data_folders, parse_folder_name_os, gen_label_os,extract_app_from_dict,parse_app_name,parse_folder_name_browser,gen_label_browser,gen_label_app_os_browser
from hcl_helpers import read_label_data
from functools import partial
from multiprocessing import Pool
import numpy as np
import pandas as pd


"""
FIX:
"""
"""
Instructions:
	1. Create a converter object
	2. activate
	3. Access / get / write data
"""
class Converter(object):
	""" FIX - Fix default feature_methods_list """
	def __init__(self, PARENT_DIRECTORY, feature_methods_list=['packet_count', 'sizemean', 'sizevar']):
		print ('Initializing...')
		print
		self.p = Pool(16)
		self.data_folders = gen_data_folders(PARENT_DIRECTORY)
		self.feature_methods = feature_methods_list
		self.all_samples = np.array([])
		print ('Done Initializing')

	"""
	Dynamically call feature methods and generate feature vector from pcap file
	"""
	def pcap_to_feature_vector(self, pcap_path, label):

		sess = Session.from_filename(pcap_path)

		feature_vector = np.array([])
		for method_name in self.feature_methods:
			if method_name!='label':
				
				method = getattr(sess, method_name)

				if not method:
					raise Exception("Method %s not implemented" % method_name)
				feature_vector = np.append(feature_vector, method())

		feature_vector = np.append(feature_vector, label)

		return feature_vector


	""" Return a list of sample feature vectors for a given child data directory """



	
	def sessions_to_samples(self, CHILD_DIRECTORY,features):
		labels = [0,1,2,3,4,5]
		samples = []
		x = 0
		for folder in CHILD_DIRECTORY:
			folder_split =folder.split('\\')
			folder_name = folder_split[len(folder_split)-1]

			only_pcap_files = gen_pcap_filenames(folder)

			for file in only_pcap_files:

				print("file: ",file)
				l =  self.pcap_to_feature_vector(file,labels[x])
			
				samples.append(l)
			x = x + 1
			

		return samples
		

	
	def activate(self,features):
		
		samples = self.sessions_to_samples(self.data_folders,features)

		self.all_samples = samples
		
		

	"""  """
	def get_samples(self):
		return self.all_samples

	""" TEST THIS ... [] operator """
	def __getitem__(self,index):
		return self.all_samples[index]


	""" TEST THIS ... return an iterator """
	def __iter__(self):
		return iter(self.all_samples)

	""" Write samples to csv """
	def write_to_csv_every_session(self,my_all_samples, file_name, column_names):
		sdf = pd.DataFrame(my_all_samples, columns=column_names)
		sdf.to_csv(file_name)
	
	
	def write_to_csv(self, file_name, column_names):


		output_filename='samples.csv',
		sdf = pd.DataFrame(self.all_samples, columns=column_names)
		sdf.to_csv(file_name)
		
	
	def write_to_csv_tab(self, file_name, separator, column_names):

		
		sdf = pd.DataFrame(self.all_samples, columns=column_names)
		sdf.to_csv(file_name,sep=separator,index=False)
		