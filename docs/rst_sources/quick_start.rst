Quick Starters Guide
====================

**Setup the Buddi-CRAVeD environment** by following the instructions on the :ref:`Setting up the Environment <setup>` page.

Use the :download:`generate_samplings.py <documents/generate_samplings.py>` and :download:`generate_indices.py <documents/generate_indices.py>` template scripts to perform **random sampling and cluster validation** of the datasets. Uncomment, comment or modify the parameter arguments in the scripts to suit your needs and run them.

The **random samplings** (bagging) and **cluster validation** (internal and external indices) results are dumped into the Buddi-CRAVeD warehouse.

==========================
Random Sampling of Dataset
==========================

The :download:`generate_samplings.py <documents/generate_samplings.py>` script performs the following tasks:

	* Reads the dataset from CSV, LIBSVM or ARFF data files
	* Preprocesses (i.e., Dummy codes and Standardizes) the data
	* Performs Repeated Random Sampling (with replacement across samplings) of the datasets into bags

The sampled data bags are dumped into the :file:`BAGS` folder in the Buddi-CRAVeD warehouse.

.. literalinclude:: documents/generate_samplings.py
	:emphasize-lines: 4,5,93,108,144,166,172,192
	:caption: generate_samplings.py

==================================
Cluster Validation on Sampled Data
==================================

The :download:`generate_indices.py <documents/generate_indices.py>` script performs the following tasks:

	* Loads the sampled data bags one after another
	* Processeses (i.e, standardizes) the bag data samples
	* Estimates Optimal Number of Clusters in data for K-Means and Hierarchial Cluster Analysis (using **GAP STATISTICS**)
	* Performs K-Means and Hierarchial Clustering of the data
	* Evaluates cluster quality by generating internal and external cluster validation indices.

The cluster validation results are appended to the :file:`results.csv` file in the Buddi-CRAVeD warehouse.

.. literalinclude:: documents/generate_indices.py
	:tab-width: 4
	:caption: generate_indices.py