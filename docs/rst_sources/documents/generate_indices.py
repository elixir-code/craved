""" Script to generate indices from randomly sampled bags of data """

from craved import eda
from craved import internal_indices
from craved import external_indices

import pickle

""" Parameters to be specified by the user 
Instruction: uncomment and modify the necessary argument lines for the parameters and comment other (default) argument lines.
"""

# path to the location where the folder with dataset's (randomly sampled) bag folder resides
location = eda.__warehouse__ + 'BAGS/' # default
# location = '<replace with path to the location>'


# name of folder which contains the dataset's (randomly sampled) bag and metadata files.
bag_name = '<replace with name of folder which contains datasets bag and metadata files>'


'''Parameters for K-Means Cluster Analysis'''

# Method used to determine the number of clusters in data for K-Means Clustering. 
# Allowed arguments: {'gap', 'n_classes', <int>, <list of integers of shape (n_bags,)>}
kmeans_n_clusters = 'gap' # Use gap statistics to estimate the optimal number of clusters from data (default)
# n_clusters = 'n_classes' # Use the number of classes in data as the number of clusters
# n_clusters = <int>
#n_clusters = <list of {int, 'gap', 'n_classes'} of shape (n_bags,)> denoting the number of clusters in each bag

kmeans_kargs = {} # Other Keyword arguments (other than 'n_clusters') to :func:`sklearn.cluster.KMeans`


'''Parameters for Ward's Agglomerative Cluster Analysis'''

# Method used to determine the number of clusters in data for Hierarchial Clustering. 
# Allowed arguments: {'gap', 'n_classes', <int>, <list of integers of shape (n_bags,)>}
hierarchial_n_clusters = 'gap' # Use gap statistics to estimate the optimal number of clusters from data (default)
# n_clusters = 'n_classes' # Use the number of classes in data as the number of clusters
# n_clusters = <int>
#n_clusters = <list of {int, 'gap', 'n_classes'} of shape (n_bags,)> denoting the number of clusters in each bag

hierarchial_kargs = {} # Other Keyword arguments (other than 'n_clusters') to :func:`sklearn.cluster.AgglomerativeClustering`


""" Code to perform cluster analysis and Validation
Note: Not to be modified by the user
"""

internal_indices_functions = {
								'WGSS':'wgss_index',
								'BGSS':'bgss_index',
								'Ball-Hall':'ball_hall_index', 
								'Banfeld-Raftery':'banfeld_raftery_index', 
								'Det-Ratio':'det_ratio_index', 
								'Ksq-DetW':'ksq_detw_index', 
								'Log-Det-Ratio':'log_det_ratio_index',
								'Log-SS-Ratio':'log_ss_ratio_index', 
								'Scott-Symons':'scott_symons_index',
								'Trace-WiB':'trace_wib_index',
								'Silhouette':'silhouette_score',
								'Calinski-Harabasz':'calinski_harabaz_score',
								'C':'c_index', 
								'Dunn':'dunn_index',
								'Davies-Bouldin':'davies_bouldin_index',
								'Ray-Turi':'ray_turi_index',
								'Hartigan':'hartigan_index',
								'PBM':'pbm_index',
								'Score':'score_function'
							}

# Note: Strictly don't modify (or reorder) the 'choosen_internal_indices' list (without corresponding changes to 'results.csv' in warehouse)
choosen_internal_indices = [
								'WGSS', 'BGSS',

								'Ball-Hall',
								'Banfeld-Raftery',
								'Calinski-Harabasz',
								
								'Det-Ratio',
								'Ksq-DetW',
								'Log-Det-Ratio',
								'Log-SS-Ratio',
								
								#'Scott-Symons',
								'Silhouette',

								'Trace-WiB',
								'C',
								'Dunn',
								'Davies-Bouldin',
								'Ray-Turi',
								'PBM',
								'Score'
							]

# TODO*: Check if Sklearn's Precision, Recall, ... which are for classification work for clustering as well (change cluster indices and observe change in precision) 
external_indices_functions = {
								'Entropy':'entropy',
								'Purity':'purity',

								'Precision':'precision_coefficient',
								'Recall':'recall_coefficient',
								'F':'f_measure',
								'Weighted-F':'weighted_f_measure',

								'Folkes-Mallows':'folkes_mallows_index',
								'Rand':'rand_index',
								'Adjusted-Rand':'adjusted_rand_index',

								'Adjusted-Mutual-Info':'adjusted_mutual_info',
								'Normalised-Mutual-Info':'normalized_mutual_info',

								'Homegeneity':'homogeneity_score',
								'Completeness':'completness_score',
								'V-Measure':'v_measure_score',

								'Jaccard':'jaccard_coeff',
								'Hubert Γ̂':'hubert_T_index',
								'Kulczynski':'kulczynski_index',

								'McNemar':'mcnemar_index',
								'Phi':'phi_index',
								'Russel-Rao':'russel_rao_index',

								'Rogers-Tanimoto':'rogers_tanimoto_index',
								'Sokal-Sneath1':'sokal_sneath_index1',
								'Sokal-Sneath2':'sokal_sneath_index2'
							}

# Note: Strictly don't modify (or reorder) the 'choosen_external_indices' list (without corresponding changes to 'results.csv' in warehouse)
choosen_external_indices = [
								'Entropy',
								'Purity',
								
								'Precision',
								'Recall',
								'F',
								'Weighted-F',

								'Folkes-Mallows',
								'Rand',
								'Adjusted-Rand',

								'Adjusted-Mutual-Info',
								'Normalised-Mutual-Info',

								'Homegeneity',
								'Completeness',
								'V-Measure',

								'Jaccard',
								'Hubert Γ̂',
								'Kulczynski',

								'McNemar',
								'Phi',
								'Russel-Rao',

								'Rogers-Tanimoto',
								'Sokal-Sneath1',
								'Sokal-Sneath2'
							]

# Extract the list of all (randomly sampled) dataset bag files in sorted order
import re, os
bag_files = sorted([file.path for file in os.scandir(location + bag_name) if file.is_file() and re.match(r"[_]?bag[0-9]+.p$", file.name)])
bag_files_names = sorted([file.name for file in os.scandir(location + bag_name) if file.is_file() and re.match(r"[_]?bag[0-9]+.p$", file.name)])

# Open and read the bags and dataset metadata info.
metadata_file = location + bag_name + "/metadata.p"
metadata = pickle.load( open(metadata_file, 'rb') )

# Open 'results.csv' file in warehouse and append all results to it.
results_file = open(eda.__warehouse__+"results.csv", 'a')

n_bags = len(bag_files)

# Preprocess the 'kmeans_n_clusters' argument to standard accepted forms
if isinstance(kmeans_n_clusters, str) and kmeans_n_clusters in ['gap', 'n_classes']:
	kmeans_n_clusters = [kmeans_n_clusters] * n_bags

elif isinstance(kmeans_n_clusters, int):
	kmeans_n_clusters = [kmeans_n_clusters] * n_bags

elif isinstance(kmeans_n_clusters, list):
	if len(kmeans_n_clusters)!=n_bags:
		print("error: The length of 'kmeans_n_clusters' list supplied doesn't match the count of the number of bags")
		sys.exit(1)

	for bag_n_cluster in kmeans_n_clusters:
		if isinstance(bag_n_cluster, int):
			pass

		elif isinstance(bag_n_cluster, 'str') and bag_n_cluster in ['gap','n_classes']:
			pass

		else:
			print("error: invalid entry %s in list supplied as argument to parameter 'kmeans_n_clusters'"%bag_n_cluster.__repr__())
			sys.exit()

else:
	print("error: invalid argument to parameter 'kmeans_n_clusters'. Accepted arguments: {'gap', 'n_classes', <int>, <list of {int,'gap','n_classes'} of shape (n_bags,)> }")


# Preprocess the 'hierarchial_n_clusters' argument to standard accepted forms
if isinstance(hierarchial_n_clusters, str) and hierarchial_n_clusters in ['gap', 'n_classes']:
	hierarchial_n_clusters = [hierarchial_n_clusters] * n_bags

elif isinstance(hierarchial_n_clusters, int):
	hierarchial_n_clusters = [hierarchial_n_clusters] * n_bags

elif isinstance(hierarchial_n_clusters, list):
	if len(hierarchial_n_clusters)!=n_bags:
		print("error: The length of 'hierarchial_n_clusters' list supplied doesn't match the count of the number of bags")
		sys.exit(1)

	for bag_n_cluster in hierarchial_n_clusters:
		if isinstance(bag_n_cluster, int):
			pass

		elif isinstance(bag_n_cluster, 'str') and bag_n_cluster in ['gap','n_classes']:
			pass

		else:
			print("error: invalid entry %s in list supplied as argument to parameter 'hierarchial_n_clusters'"%bag_n_cluster.__repr__())
			sys.exit()

else:
	print("error: invalid argument to parameter 'hierarchial_n_clusters'. Accepted arguments: {'gap', 'n_classes', <int>, <list of {int,'gap','n_classes'} of shape (n_bags,)> }")


for bag_index, bag_file in enumerate(bag_files):

	print("Processing '{file_name}' ({bag_index}/{n_bags})".format(file_name=bag_files_names[bag_index], bag_index=bag_index+1, n_bags=n_bags))

	dataset = pickle.load( open(bag_file, 'rb') ) 
	data, target = dataset['data'], dataset['target']

	# Load the data into an eda object :obj:`main`
	main = eda.eda()
	main.load_data(data, target)

	# perform kmeans clustering on the data
	main.perform_kmeans(n_clusters=kmeans_n_clusters[bag_index], **kmeans_kargs)

	# Compute the internal indices for K-Means Clustering Results
	kmeans_internal_validation = internal_indices.internal_indices(main.data, main.kmeans_results['labels'])

	kmeans_internal_indices = []

	for index in choosen_internal_indices:
		index_function = getattr(kmeans_internal_validation, internal_indices_functions[index])
		print("Computing Internal Indices (KMeans Clustering): %s Index"%index)
		kmeans_internal_indices.append(index_function())

	# Compute the external indices for K-Means Clustering Results
	kmeans_external_validation = external_indices.external_indices(main.target, main.kmeans_results['labels'])

	kmeans_external_indices = []

	for index in choosen_external_indices:
		index_function = getattr(kmeans_external_validation, external_indices_functions[index])
		print("Computing External Indices (KMeans Clustering): %s Index"%index)
		kmeans_external_indices.append(index_function())


	# Print the KMeans Clustering results to a 'results.csv' in warehouse
	results_file.write("\"{dataset}\",{timestamp_id},{n_samples},{n_features},{n_classes},\"{class_distrn}\",{nominal_features},".format(dataset=bag_name, timestamp_id=metadata['timestamp'], n_samples=metadata['n_samples'], n_features=metadata['n_features'], n_classes=len(metadata['classes']), class_distrn=metadata['classes'].__repr__(), nominal_features="No" if all(value is None for value in metadata['column_categories'].values()) else "Yes"))
	results_file.write("\"{file_name}\",{sample_size},\"{algorithm}\",\"{parameters}\",{n_clusters},\"{cluster_distrn}\",".format(file_name=bag_files_names[bag_index], sample_size=main.n_samples, algorithm='KMeans', parameters=main.kmeans_results['parameters'].__repr__(), n_clusters=main.kmeans_results['n_clusters'], cluster_distrn=main.kmeans_results['clusters'].__repr__()))

	for index in kmeans_internal_indices:
		results_file.write("{},".format(index))

	for index in kmeans_external_indices:
		results_file.write("{},".format(index))

	results_file.write('\n')

	# perform Ward's Agglomerative clustering on the data
	main.perform_hierarchial(n_clusters=hierarchial_n_clusters[bag_index], **hierarchial_kargs)

	# Compute the internal indices for K-Means Clustering Results
	hierarchial_internal_validation = internal_indices.internal_indices(main.data, main.hierarchial_results['labels'])

	hierarchial_internal_indices = []

	for index in choosen_internal_indices:
		index_function = getattr(hierarchial_internal_validation, internal_indices_functions[index])
		print("Computing Internal Indices (Hierarchial Clustering): %s Index"%index)
		hierarchial_internal_indices.append(index_function())

	# Compute the external indices for Hierarchial Clustering Results
	hierarchial_external_validation = external_indices.external_indices(main.target, main.hierarchial_results['labels'])

	hierarchial_external_indices = []

	for index in choosen_external_indices:
		index_function = getattr(hierarchial_external_validation, external_indices_functions[index])
		print("Computing External Indices (Hierarchial Clustering): %s Index"%index)
		hierarchial_external_indices.append(index_function())

	# Print the Hierarchial Clustering results to a 'results.csv' in warehouse
	results_file.write("\"{dataset}\",{timestamp_id},{n_samples},{n_features},{n_classes},\"{class_distrn}\",{nominal_features},".format(dataset=bag_name, timestamp_id=metadata['timestamp'], n_samples=metadata['n_samples'], n_features=metadata['n_features'], n_classes=len(metadata['classes']), class_distrn=metadata['classes'].__repr__(), nominal_features="No" if all(value is None for value in metadata['column_categories'].values()) else "Yes"))
	results_file.write("\"{file_name}\",{sample_size},\"{algorithm}\",\"{parameters}\",{n_clusters},\"{cluster_distrn}\",".format(file_name=bag_files_names[bag_index], sample_size=main.n_samples, algorithm='Hierarchial', parameters=main.hierarchial_results['parameters'].__repr__(), n_clusters=main.hierarchial_results['n_clusters'], cluster_distrn=main.hierarchial_results['clusters'].__repr__()))

	for index in hierarchial_internal_indices:
		results_file.write("{},".format(index))

	for index in hierarchial_external_indices:
		results_file.write("{},".format(index))

	results_file.write('\n')
	del main

	print('\n\n')

results_file.close()