import numpy as np
import sys

from sklearn.cluster import KMeans
def gap_statistics(data, kmax=10, kmin=1, B=10, cluster_algorithm=KMeans, **kargs):

	if kmin>kmax:
		print("error: invalid arguments (kmin=%d and kmax=%d) to parameters 'kmin' and 'kmax' for gap statistics."%(kmin, kmax))
		sys.exit(1)

	n_samples, n_features = data.shape
	K = range(kmin, kmax+2)

	logwks = np.empty(len(K))

	# Cluster original data with n_clusters in range(kmin, kmax+2) and compute corresponding log(wk) values.
	for index, n_clusters in enumerate(K):
		clusterer = cluster_algorithm(n_clusters=n_clusters, **kargs)
		labels = clusterer.fit_predict(data)

		# For KMeans Clustering (optimisation)
		if hasattr(clusterer, 'inertia_'):
			wk = clusterer.inertia_

		else:
			wk = compute_wk(data, labels)

		logwks[index] = np.log(wk)
		#print("info: For Data, Completed K=%d"%n_clusters)

	ref_data_logwks = np.empty(shape=(B, len(K)))
	maximum, minimum = np.max(data, axis=0), np.min(data, axis=0)

	# Cluster B reference data with n_clusters in range(kmin, kmax+2) and compute corresponding log(wk) values.
	# Note: method (a) of generating reference data uniformly from bounding box is used (for simplicity)
	for b in range(B):
		ref_data = np.random.random_sample(size=(n_samples, n_features))
		ref_data = ref_data*(maximum - minimum) + minimum

		for index, n_clusters in enumerate(K):
			clusterer = cluster_algorithm(n_clusters=n_clusters, **kargs)
			labels = clusterer.fit_predict(ref_data)

			# For KMeans Clustering (optimisation)
			if hasattr(clusterer, 'inertia_'):
				wk = clusterer.inertia_

			else:
				wk = compute_wk(ref_data, labels)

			ref_data_logwks[b][index] = np.log(wk)
		
		print("info: Completed for reference data, B=%d"%(b+1))

	ref_data_mean_logwks = np.mean(ref_data_logwks, axis=0)
	ref_data_logwks_sk = np.sqrt(1+1/B)*np.std(ref_data_logwks, axis=0)

	gaps_k = ref_data_mean_logwks - logwks
	optimal_k_criteria = (gaps_k[:-1] - gaps_k[1:] + ref_data_logwks_sk[1:]) >= 0

	# Return the smallest value of k for which the optimal_k_criteria is 'True' (i.e, positive)
	if optimal_k_criteria.any():
		return K[optimal_k_criteria.argmax()]

	else:
		#print("error: Failed to find 'optimal number of clusters' (k) value in given search space for k (i.e., kmin=%d to kmax=%d)."%(kmin, kmax))
		return -1

def compute_wk(data, labels):

	wk = 0
	cluster_indices = np.unique(labels)

	for cluster_index in cluster_indices:
		cluster_data_indices = np.where(labels==cluster_index)[0]
		
		if cluster_data_indices.size==0:
			continue

		cluster_data = data[cluster_data_indices]
		cluster_mean = np.mean(cluster_data, axis=0)

		wk += np.sum(np.power((cluster_data - cluster_mean),2))

	return wk


