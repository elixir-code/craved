"""Exploratory Data Analysis (EDA) Toolkit

The :mod:`craved.eda` module provides interfaces for :

* reading datasets from files (supported file-formats: **csv, libsvm, arff**)
* pre-processing datasets (**feature scaling**, **one-hot encoding** of categorical features)
* **random sampling** of datasets
* **cluster analysis** and parameter determination (supported algorithms: **K-Means, DBSCAN, HDBSCAN, Hierarchial, Spectral**)
* **data visualisation**

Attributes:
	__warehouse__ (str): The warehouse directory path, if ``warehouse`` is setup. :obj:`None` otherwise

Todo:
	* add support for libsvm, arff data formats
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans

from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from math import pow,floor,ceil


# For stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit

from . import gap

import pandas as pd
from sklearn.datasets import load_svmlight_file
from scipy.io.arff import loadarff

from sklearn.preprocessing import LabelEncoder,StandardScaler

#from itertools import combinations
import matplotlib.patches as mpatches

import h5py
import pickle

from time import time
#import warnings

from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian

import os, sys


def get_warehouse():
	"""Retrieves the **data warehouse** directory path

	Returns:
		Warehouse directory path, if ``warehouse`` is setup. :obj:`None` otherwise
	"""
	warehouse_path = None

	try:
		eda_dir_path = os.path.dirname(os.path.abspath(__file__))		
		with open(eda_dir_path + "/craved_warehouse.dat","r") as config_file:
			warehouse_path = config_file.read().strip()

	except:
		pass

	return warehouse_path

#module attribute that stores the warehouse directory path
__warehouse__ = get_warehouse()

class eda:
	"""A data container class with methods for data pre-processing and cluster analysis related tasks"""

	def __init__(self, force_file=False, filename=None, location=__warehouse__+"HOOD/" if __warehouse__!=None else None):
		"""
		Parameters:
			force_file (bool, default=False):	stores and processes memory-critical data structures on HDF5 files, if set ``True``.
			filename (str):	filename (without ``.hdf5`` extension) to use for the HDF5 file. (ignored if ``force_file=False``)
			location (str, default= ``<warehouse directory>/HOOD/`` ):	path of the **directory** to store the HDF5 file (ignored if ``force_file=False``)
		"""

		self._hdf5_file = None
		self.hood_filename = None

		if force_file:

			if not(isinstance(location, str) and isinstance(filename, str)):
				print("error : invalid location or filename specified for under the HOOD file !")
				sys.exit(1)

			if not location.endswith('/'):
				location = location + '/'

			if not (filename.endswith('.hdf5') or filename.endswith('.h5')):
				filename = filename + ".hdf5"

			self.hood_filename = location+filename

			try:
				#Note: default file mode : 'a' (Read/write if exists, create otherwise)
				self._hdf5_file = h5py.File(self.hood_filename, libver='latest')

			except Exception as err:
				print("\nerror: {0}\nfailed to create under the HOOD file !".format(err))
				sys.exit(1)


	def load_data(self, data, target=None, flatten_features=True):
		"""Load obj:`np.ndarray` or :obj:`list` objects as data and target values
		
		Parameters:
			data (:obj:`np.ndarray`): array of data samples (samples x features)
			target (:obj:`np.ndarray`, optional): class labels or target vales
			flatten_features (bool): flatten complex **multi-dimensional** features, if ``True``

		Note:
			* Complex 'multi-dimensional' features of data samples are implicitly flattened by default.
			* Column indices (or names) of the features are zero-indexed.

		Examples:

			Illustration of implicit flattening of multi-dimensional features::
			
			>>> from craved import eda
			>>> #create dummy data with multi-dimensional features
			>>> data = 	[
			...				[
			...					[[1],[2]], [[3,4],[5,6]]
			...				],
			...				[
			...					[[7],[8]], [[9,10],[11,12]]
			...				]
			...			]
			>>> main = eda.eda(force_file=False)
			>>> main.load_data(data)
			>>> print(main.data)
			>>> print("no. of samples = ", main.n_samples)
			>>> print("no. of features = ", main.n_features)
		"""
		
		try:
			data = np.array(data)

			if flatten_features:

				#Flatten 'simple' numerical multi-dimensional features
				if issubclass(data.dtype.type, np.integer) or issubclass(data.dtype.type, np.floating):
					if len(data.shape)==1:
						data = data.reshape(data.shape[0], 1)

					if len(data.shape)>2:
						data = data.reshape(data.shape[0], np.product(data.shape[1:]))

				#Flatten 'complex' non-numerical multi-dimensional features
				elif issubclass(data.dtype.type, np.object_):
					
					flattened_data = []

					for sample in data:
						flattened_data.append(flatten_list(sample))

					data = np.array(flattened_data, dtype=np.number)

					if not(issubclass(data.dtype.type, np.integer) or issubclass(data.dtype.type, np.floating)):
						raise UserWarning("error: Data contains 'non-numerical features' or 'varying number of features across samples'")

		except Exception as err:
			print('{0}\nerror: failed to load data or flatten multi-dimensional features'.format(err))
			sys.exit(1)
		
		self.data = data
		self.n_samples, self.n_features = self.data.shape

		self.columns_ =	np.arange(self.n_features)

		if target is not None:

			try:
				if self.n_samples == len(target):
					self.target = np.array(target)

				else:
					raise UserWarning("number of 'target' values doesn't match number of samples in data")

				if len(self.target.shape)>1:
					raise UserWarning("'target' values form a multi-dimensional array (but one-dimensional array expected).")

			except Exception as err:
				print('{0}\nerror: invalid target array supplied'.format(err))
				sys.exit(1)

			self.classes_ = None

			classes_ = np.unique(self.target)
			if classes_.shape[0] <= max_classes_nominal(self.n_samples):
				self.classes_ = classes_

	"""Reading datasets from standard file formats (Supported File Formats : csv, libsvm, arff)
	
		See also:
			`Loading from External datasets <http://scikit-learn.org/stable/datasets/#loading-from-external-datasets>`_
	"""

	def read_data_csv(self, file, sep=',', skiprows=None, header_row=None, usecols=None, target_col=-1, encode_target=True, categorical_cols='infer', na_values=None, nrows=None, **kargs):
		"""Read data from CSV format file
		
		Parameters:
			file (str or open file): path to the CSV data file or URL (http, ftp, S3 location) or ``open file`` object.
			sep (str, default=','): Column delimiter. Accepted values: ``None`` implies autodetect delimiter, '\s+' uses combination of spaces and tabs, Regular expressions
			
			skiprows (:obj:`list` or int, default= ``None``): 'List' (list) of line indices to skip or 'Number' (int) of starting lines to skip.
			header_row (int, default=``None``): Relative Zero-Index (index of rows after skipping rows using ``skiprows`` parameter) of the row containing column names. Note: All preceding rows are ignored.
			
			usecols (:obj:`list`, default=  ``None``): List of column 'names' (or 'indices', if no column names) to consider. ``None`` indicates use of all columns.
			target_col (int, default=``-1``): Relative Zero-Index of column (after filtering columns using ``usecols`` parameter) to use as target values. ``None`` indicates absence of target value columns.
			
			encode_target (bool, default=True): Encode target values
			categorical_cols (:obj:`list`, str, int, 'all', None, default='infer'): List (str or int if singleton) of column 'names' (or absolute 'indices', if no column names) of categorical columns to encode. ``categorical_cols='infer'`` autodetects nominal categorical columns. ``categorical_cols='all'`` implies all columns are nominal categorical. ``categorical_cols=None`` implies no nominal categorical columns.
			
			na_values (scalar, str, list-like, or dict, default=``None``): Additional strings to recognize as NA/NaN. If dict passed, specific per-column NA values. By default the following values are interpreted as NaN: ‘’, ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’, ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘N/A’, ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’.
			nrows (int, default=``None``): Number of rows of data to read. `None`` implies all available rows.
			
			**kargs: 	Other keyword arguments accepted by :func:`pandas.read_csv` (Keyword Arguments: comment, lineterminator, ...)

		Note:
			* ``skiprows`` parameter uses absolute row indices whereas ``header_row`` parameter uses relative index (i.e., zero-index after removing rows specied by ``skiprows`` parameter).
			* ``usecols`` and ``categorical_cols`` parameters use absolute column 'names' (or 'indices' if no 'names') whereas ``target_cols`` parameter uses relative column 'indices' (or 'names') after filtering out columns specified by ``usecols`` parameter.
			* ``categorical_cols='infer'`` identifies and encodes nominal features (i.e., features of 'string' type, with fewer unique entries than a value heuristically determined from number of data samples) and drops other 'string' and 'date' type features.
				use func:`craved.eda.max_classes_nominal` to find the heuristically determined value of maximum number of distinct entries in nominal features for given number of samples
			* Data samples with any NA/NaN features are implicitly dropped.

		Examples:
			Illustration of **Reading from CSV data file** ::
			
			>>> from craved import eda
			>>> main = eda.eda()
			>>>
			>>> from io import StringIO
			>>>
			>>> data = 	'''Dataset: Abalone
			...	Source: UCI ML Repository
			...	
			...	skips rows until this, i.e., skiprows = 4. Header row follows immediately, i.e., header_row = 0.
			...	Sex, Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight, Rings
			...	M,0.455,0.365,0.095,0.514,0.2245,0.101,0.15,15
			...	M,0.35,0.265,0.09,0.2255,0.0995,0.0485,0.07,7
			...	F,0.53,0.42,0.135,0.677,0.2565,0.1415,0.21,9
			...	M,0.44,0.365,0.125,0.516,0.2155,0.114,0.155,10
			... I,0.33,0.255,0.08,0.205,0.0895,0.0395,0.055,7
			... I,0.425,0.3,0.095,0.3515,0.141,0.0775,0.12,8
			...	F,0.53,0.415,0.15,0.7775,0.237,0.1415,0.33,20
			...	F,0.545,0.425,0.125,0.768,0.294,0.1495,0.26,16
			...	M,0.475,0.37,0.125,0.5095,0.2165,0.1125,0.165,9
			...	F,0.55,0.44,0.15,0.8945,0.3145,0.151,0.32,19
			...	'''
			>>>
			>>> # use columns ['Sex', 'Length', 'Diameter', 'Height', 'Rings']. 'Ring' is the target to predict, i.e., target_col=-1 .
			...	# Auto-detect nominal categorical columns to encode, i.e., categorical_cols='infer' (default)
			... main.read_data_csv(StringIO(data), sep=',', skiprows=4, header_row=0, usecols=['Sex', 'Length', 'Diameter', 'Height', 'Rings'], target_col=-1, encode_target=False)
			>>>
			>>> # Print the processed data samples. Note: 'Sex' column has been encoded.
			... print(main.data)
			[[ 2.     0.455  0.365  0.095]
			 [ 2.     0.35   0.265  0.09 ]
			 [ 0.     0.53   0.42   0.135]
			 [ 2.     0.44   0.365  0.125]
			 [ 1.     0.33   0.255  0.08 ]
			 [ 1.     0.425  0.3    0.095]
			 [ 0.     0.53   0.415  0.15 ]
			 [ 0.     0.545  0.425  0.125]
			 [ 2.     0.475  0.37   0.125]
			 [ 0.     0.55   0.44   0.15 ]]
			>>>
			>>> # Print the names of columns in data
			... print(main.columns_)
			Index(['Sex', 'Length', 'Diameter', 'Height'], dtype='object')
			>>>
			>>> # Print the target values, i.e, 'Rings' values.
			... print(main.target)
			[15  7  9 10  7  8 20 16  9 19]


			::
			
			>>> from craved import eda
			>>> main = eda.eda()
			>>>
			>>> from io import StringIO
			>>>
			>>> # First 10 samples from Dataset : Mushroom (UCI ML Repository). A string type feature was intentionally introduced as Column '0'.
			>>> data = 	'''
			...	sample1     p x s n t p f c n k e e s s w w p w o p k s u
			...	sample2     e x s y t a f c b k e c s s w w p w o p n n g
			...	sample3     e b s w t l f c b n e c s s w w p w o p n n m
			... sample4     p x y w t p f c n n e e s s w w p w o p k s u
			...	sample5     e x s g f n f w b k t e s s w w p w o e n a g
			...	sample6     e x y y t a f c b n e c s s w w p w o p k n g
			...	sample7     e b s w t a f c b g e c s s w w p w o p k n m
			...	sample8     e b y w t l f c b n e c s s w w p w o p n s m
			...	sample9     p x y w t p f c n p e e s s w w p w o p k v g
			...	sample10    e b s y t a f c b g e c s s w w p w o p k s m
			...	'''
			>>>
			>>> # Column delimiter is spaces or tabs, i.e., sep='\s+'
			... # No header rows available, i.e., header_row=None (default).
			... # Use all columns, i.e., usecols=None (default).
			... # Column '1' contains target values. Encode the target values, i.e., encode_target=True (default).
			... main.read_data_csv(StringIO(data), sep='\s+', header_row=None, target_col=1)
			info: columns  [0] was/were inferred as 'string' or 'date' type feature(s) and dropped
			>>>
			>>> #Print the processed data samples. Note: Column '0' was inferred as 'string' type feature and dropped.
			... print(main.data)
			[[ 1.  0.  1.  1.  3.  0.  0.  1.  1.  0.  1.  0.  0.  0.  0.  0.  0.  0.	1.  0.  2.  2.]
			 [ 1.  0.  3.  1.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.   1.  1.  1.  0.]
			 [ 0.  0.  2.  1.  1.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.  0.  0.   1.  1.  1.  1.]
			 [ 1.  1.  2.  1.  3.  0.  0.  1.  2.  0.  1.  0.  0.  0.  0.  0.  0.  0.   1.  0.  2.  2.]
			 [ 1.  0.  0.  0.  2.  0.  1.  0.  1.  1.  1.  0.  0.  0.  0.  0.  0.  0.   0.  1.  0.  0.]
			 [ 1.  1.  3.  1.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.  0.  0.   1.  0.  1.  0.]
			 [ 0.  0.  2.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.   1.  0.  1.  1.]
			 [ 0.  1.  2.  1.  1.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.  0.  0.   1.  1.  2.  1.]
			 [ 1.  1.  2.  1.  3.  0.  0.  1.  3.  0.  1.  0.  0.  0.  0.  0.  0.  0.   1.  0.  3.  0.]
			 [ 0.  0.  3.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.   1.  0.  2.  1.]]
			>>>
	    	>>> # Print the names of columns in data
			... print(main.columns_)
			Int64Index([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,	21, 22, 23], dtype='int64')
			>>>
			>>> # Print the target values, i.e, Column '1' values.
			... print(main.target)
			[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
			>>>
			>>> # Print the distinct (original) classes in target values
			... print(main.classes_)
			['e', 'p']
		"""
		dataset = pd.read_csv(filepath_or_buffer=file, sep=sep, skiprows=skiprows, header=header_row, usecols=usecols, index_col=target_col, na_values=na_values, skipinitialspace=True, nrows=nrows, **kargs)
		dataset.dropna(axis='index', how='any', inplace=True)

		# column index (or names) in data
		self.columns_ = dataset.columns
		columns_dtypes = dataset.dtypes.values

		data, target = dataset.values, None if target_col is None else np.array(dataset.index)
		del dataset

		if target is not None:

			# Distinct (original) classes in target values
			self.classes_ = None
			
			if encode_target:
				target_labelEncoder = LabelEncoder()
				target = target_labelEncoder.fit_transform(target)
				self.classes_ = target_labelEncoder.classes_.tolist()
				del target_labelEncoder

		# Column name indexed dictionary of distinct (original) categories in the data columns. Defaults to ``None`` for numeric (non-categorical) valued columns.
		self.columns_categories_ = dict.fromkeys(self.columns_)

		# using array of absolute (zero-)indices of columns for ``catergorical_cols`` parameter
		if isinstance(categorical_cols, str) and categorical_cols.casefold()=="infer":

			n_samples, n_features = data.shape
			selected_columns = np.array([True]*n_features)

			# maximum number of classes in a column to be "infered" as "categorical (nominal)"
			max_infer_nominal_classes = max_classes_nominal(n_samples)

			self._nominal_columns = []

			for column_index in np.where(columns_dtypes==np.object)[0]:

				column_labelEncoder = LabelEncoder()
				column_labelEncoder.fit(data.T[column_index])

				if len(column_labelEncoder.classes_) <= max_infer_nominal_classes:
					self._nominal_columns.append(self.columns_[column_index])
					self.columns_categories_[self.columns_[column_index]] = column_labelEncoder.classes_.tolist()
					data.T[column_index] = column_labelEncoder.transform(data.T[column_index])

				else:
					selected_columns[column_index] = False
					del self.columns_categories_[self.columns_[column_index]]

				del column_labelEncoder

			if self._nominal_columns:
				print("info: columns ",self._nominal_columns,"was/were inferred as 'nominal' categorical feature(s) and encoded")

			if not selected_columns.all():
				print("info: columns ",self.columns_[np.where(selected_columns==False)].tolist(),"was/were inferred as 'string' or 'date' type feature(s) and dropped")

			self.columns_ = self.columns_[selected_columns]
			data = data.T[selected_columns].T

		elif isinstance(categorical_cols, str) and categorical_cols.casefold()=='all':

			self._nominal_columns = self.columns_.copy()

			for column_index in range(self.columns_.shape[0]):
				column_labelEncoder = LabelEncoder()
				data.T[column_index] = column_labelEncoder.fit_transform(data.T[column_index])
				self.columns_categories_[self.columns_[column_index]] = column_labelEncoder.classes_.tolist()
				del column_labelEncoder

		elif isinstance(categorical_cols, list) or isinstance(categorical_cols, int) or isinstance(categorical_cols, str):

			if isinstance(categorical_cols, int) or isinstance(categorical_cols, str):
				categorical_cols = [categorical_cols]

			self._nominal_columns = categorical_cols.copy()

			# TODO: Process each column in a seperate thread
			for column_name in categorical_cols:	
				
				column_index, = np.where(self.columns_==column_name)
				
				if column_index.shape == (1,):
					column_labelEncoder = LabelEncoder()
					data.T[column_index[0]] = column_labelEncoder.fit_transform(data.T[column_index[0]])
					self.columns_categories_[column_name] = column_labelEncoder.classes_.tolist()
					del column_labelEncoder

				else:
					print("warning: column '{0}' could not be (uniquely) identified and was skipped".format(column_name))
					self._nominal_columns.remove(column_name)
					continue

		elif categorical_cols is None:
			self._nominal_columns = None

		else:
			print("error: Invalid argument for parameter 'categorical_cols'. Accepted arguments: {list of names (or indices) of nominal columns, 'infer', 'all', None}")
			sys.exit(1)

		try:
			data = data.astype(np.number)

		except ValueError as err:
			print("warning: Data contains 'string' (or 'date') type features and could not be casted to 'numerical' type")

		self.data, self.target = data, target
		self.n_samples, self.n_features = self.data.shape

	def read_data_libsvm(self, file, type='classification', dtype=np.float, n_features=None, **kargs):
		"""Read data from LIBSVM format file

		Parameters:
			file (str or open file or int): Path to LIBSVM data file or ``open file`` object or file descriptor
			type ({'classification','regression','ranking'}, default='classification'): Type of dataset
			dtype (datatypes, default=``np.float``): Datatype of data array
			n_features (int, default= ``None``): Number of features to use. ``None`` implies infer from data.

			**kargs: 	Other Keyword arguments accepted by :func:`sklearn.datasets.load_svmlight_file` (Keyword arguments : offset, length, multilabel ...)
		
		Note:
			* ``file-like`` objects passed to 'file' parameter must be opened in binary mode.
			* Learning to Rank('ranking' type) datasets are not currently supported
			* ``dtype`` parameter accepts only numerical datatypes
			* The LIBSVM data file is assumed to have been already preprocessed, i.e., encoding categorical features and removal of missing values.

		Examples:
			Illustration of **Reading from LIBSVM data file** ::

			>>> from craved import eda
			>>> main = eda.eda()
			>>>
			>>> from io import BytesIO
			>>>
			>>> # First 10 samples from dataset Breast Cancer (Source: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/breast-cancer)
			... data = b'''
			... 2.000000  1:1000025.000000 2:5.000000 3:1.000000 4:1.000000 5:1.000000 6:2.000000 7:1.000000 8:3.000000 9:1.000000 10:1.000000
			...	2.000000  1:1002945.000000 2:5.000000 3:4.000000 4:4.000000 5:5.000000 6:7.000000 7:10.000000 8:3.000000 9:2.000000 10:1.000000
			...	2.000000  1:1015425.000000 2:3.000000 3:1.000000 4:1.000000 5:1.000000 6:2.000000 7:2.000000 8:3.000000 9:1.000000 10:1.000000
			...	2.000000  1:1016277.000000 2:6.000000 3:8.000000 4:8.000000 5:1.000000 6:3.000000 7:4.000000 8:3.000000 9:7.000000 10:1.000000
			...	2.000000  1:1017023.000000 2:4.000000 3:1.000000 4:1.000000 5:3.000000 6:2.000000 7:1.000000 8:3.000000 9:1.000000 10:1.000000
			...	4.000000  1:1017122.000000 2:8.000000 3:10.000000 4:10.000000 5:8.000000 6:7.000000 7:10.000000 8:9.000000 9:7.000000 10:1.000000
			...	2.000000  1:1018099.000000 2:1.000000 3:1.000000 4:1.000000 5:1.000000 6:2.000000 7:10.000000 8:3.000000 9:1.000000 10:1.000000
			...	2.000000  1:1018561.000000 2:2.000000 3:1.000000 4:2.000000 5:1.000000 6:2.000000 7:1.000000 8:3.000000 9:1.000000 10:1.000000
			...	2.000000  1:1033078.000000 2:2.000000 3:1.000000 4:1.000000 5:1.000000 6:2.000000 7:1.000000 8:1.000000 9:1.000000 10:5.000000
			...	2.000000  1:1033078.000000 2:4.000000 3:2.000000 4:1.000000 5:1.000000 6:2.000000 7:1.000000 8:2.000000 9:1.000000 10:1.000000
			...	'''
			>>>
			>>> import numpy as np
			>>> # Each row is an instance and takes the form **<target value> <feature index>:<feature value> ... **.
			...	# Dataset is 'classification' type and target values (first column) represents class label of each sample, i.e., type='classification' (default)
			...	# All features assume only integral values, i.e., dtype=np.int
			...	main.read_data_libsvm(BytesIO(data), dtype=np.int)
			>>>
			>>> # Print the data samples
			... print(main.data)
			[[1000025       5       1       1       1       2       1       3       1       1]
			 [1002945       5       4       4       5       7      10       3       2       1]
			 [1015425       3       1       1       1       2       2       3       1       1]
			 [1016277       6       8       8       1       3       4       3       7       1]
			 [1017023       4       1       1       3       2       1       3       1       1]
			 [1017122       8      10      10       8       7      10       9       7       1]
			 [1018099       1       1       1       1       2      10       3       1       1]
			 [1018561       2       1       2       1       2       1       3       1       1]
			 [1033078       2       1       1       1       2       1       1       1       5]
			 [1033078       4       2       1       1       2       1       2       1       1]]
			>>>
			>>> # Print indices of columns or features. Assumption: Feature indices always uses one-based index
			...	print(main.columns_)
			[ 1  2  3  4  5  6  7  8  9 10]
			>>>
			>>> # Print target values
			... print(main.target)
			[2 2 2 2 2 4 2 2 2 2]
			>>>
			>>> # Print the distinct classes in target values
			...	print(main.classes_)
			[2 4]
		"""
		dataset = load_svmlight_file(f=file, dtype=dtype, n_features=n_features, query_id=False, **kargs)
		data, target = dataset[0].toarray(), dataset[1]

		del dataset

		self.classes_ = None

		if type.casefold()=="classification":
			target = target.astype(np.int)
			self.classes_ = np.unique(target)

		elif type.casefold()=="regression":
			pass

		elif type.casefold()=="ranking":
			print("error: 'ranking' type datasets are not currently supported")
			sys.exit(1)

		n_features = data.shape[1]
		self.columns_ = np.arange(1, n_features+1)

		self._nominal_columns = None

		self.data, self.target = data, target
		self.n_samples, self.n_features = self.data.shape

	# TODO: Allow use of subset of attributes
	def read_data_arff(self, file, target_attr='class', encode_target='infer', num_categorical_attrs=None, drop_na_rows=True):
		"""Read data from ARFF format file
		
		Parameters:
			file (str or open file): path to ARFF data file or ``open file`` object
			
			target_attr (str, default='class'): attribute name of the target column. ``target_attr=None``implies no target columns.
			encode_target (bool, default-'infer'): Encode target values. ``encode_target='infer'`` encodes nominal target and ignores numeric target attributes.

			num_categorical_attrs (:obj:`list`, default= ``None``): List of 'names' of numeric attributes to be inferred as nominal and to be encoded. Note: All nominal attributes are implicitly encoded.
			drop_na_rows (bool, detault=True): Drop data samples with NA/NaN ('?') features
		
		Note:
			All nominal type attributes are implicitly encoded.

		Examples:
			Illustration of **Reading from ARFF data file** ::

			>>> from craved import eda
			>>> main = eda.eda()
			>>>
			>>> from io import StringIO
			>>>
			>>> # An excerpt from dataset 'Hepatitis' involving features 'Age', 'Sex', 'Steroid', Albumin', 'Protime' and 'Class'.
			>>> data = '''
			...	% Dataset: Hepatitis (Source: Weka)
			... @relation hepatitis
			...	
			...	@attribute Age integer
			...	@attribute Sex {male, female}
			...	@attribute Steroid {no, yes}
			...	@attribute Albumin real
			...	@attribute Class {DIE, LIVE}
			...
			...	@data
			...	30,male,no,4,LIVE
			...	50,female,no,3.5,LIVE
			...	78,female,yes,4,LIVE
			...	31,female,?,4,LIVE
			...	34,female,yes,4,LIVE
			...	46,female,yes,3.3,DIE
			...	44,female,yes,4.3,LIVE
			...	61,female,no,4.1,LIVE
			...	53,male,no,4.1,LIVE
			...	43,female,yes,3.1,DIE
			...	'''
			>>>
			>>> # The target is attribute 'Class', i.e., target_attr='Class'
			...	# Data samples with any missing ('?') features should be dropped, i.e., drop_na_rows=True (default).
			... main.read_data_arff(StringIO(data), target_attr='Class')
			info: The dataset may contain attributes with N/A ('?') values
			>>>
			>>> # Print the processed data samples.
			...	'''Note:	Nominal features ['Sex', 'Steroid'] have been implicitly encoded.
			...				Samples with any missing value('?') features have been dropped'''
			[[ 30.    1.    0.    4. ]
			 [ 50.    0.    0.    3.5]
			 [ 78.    0.    1.    4. ]
			 [ 34.    0.    1.    4. ]
			 [ 46.    0.    1.    3.3]
			 [ 44.    0.    1.    4.3]
			 [ 61.    0.    0.    4.1]
			 [ 53.    1.    0.    4.1]
			 [ 43.    0.    1.    3.1]]
			>>>
			>>> # Print the names of columns in data
			... print(main.columns_)
			['Age', 'Sex', 'Steroid', 'Albumin']
			>>>
			>>> # Print the target values. Note: Target attribute 'Class' has been encoded.
			...	print(main.target)
			[1 1 1 1 0 1 1 1 0]
			>>>
			>>> # Print the distinct (original) classes in target values
			... print(main.classes_)
			['DIE', 'LIVE']
		"""
		dataset, metadata = loadarff(f=file)

		rows_without_na = np.ones(dataset.shape[0], dtype=np.bool)
			
		for attribute in metadata:	
			if metadata[attribute][0] == 'nominal':
				rows_without_na[np.where(dataset[attribute] == b'?')] = False

			if metadata[attribute][0] == 'numeric':
				rows_without_na[np.isnan(dataset[attribute])] = False
		
		if not rows_without_na.all():
			print("info: The dataset may contain attributes with N/A ('?') values")

		if drop_na_rows:
			dataset = dataset[rows_without_na]

		# if target_attr is None or target_attr in metadata:
		# 	data_records, target = dataset[[attribute for attribute in metadata if attribute!=target_attr]], None if target_attr is None else dataset[target_attr]

		self.columns_ = metadata.names().copy()

		if target_attr is None or target_attr in metadata:
	
			if target_attr in metadata:
				self.columns_.remove(target_attr)

			data_records, target = dataset[self.columns_], None if target_attr is None else dataset[target_attr]			
			del dataset

		else:
			print("error: Unknown 'target' attribute name specified")
			sys.exit(1)

		# Processing target labels
		if target_attr is not None:

			self.classes_ = None

			# 'classification' type datasets
			if metadata[target_attr][0]=='nominal':
				if isinstance(encode_target, str) and encode_target.casefold()=='infer':
					encode_target = True

			# 'regression' type datasets
			elif metadata[target_attr][0]=='numeric':
				target = target.astype(np.number)
				if isinstance(encode_target, str) and encode_target.casefold()=='infer':
					encode_target = False

			if encode_target:
				target_labelEncoder = LabelEncoder()
				target = target_labelEncoder.fit_transform(target)
				self.classes_ = [target_class.decode() for target_class in target_labelEncoder.classes_.tolist()]
				#self.classes_ = target_labelEncoder.classes_.tolist()
		
		# Form a new data array
		data = np.empty( ( data_records.size, len(data_records.dtype.names) ), dtype=np.float64)

		self._nominal_columns = []

		# Column name indexed dictionary of distinct (original) categories in the data columns. Defaults to ``None`` for numeric (non-categorical) valued columns.
		self.columns_categories_ = dict.fromkeys(self.columns_)

		for index, attribute in enumerate(data_records.dtype.names):

			attribute_values = data_records[attribute]
			encode_attribute = False

			if metadata[attribute][0] == 'numeric':

				if num_categorical_attrs is not None and attribute in num_categorical_attrs:
					encode_attribute = True

			elif metadata[attribute][0] == 'nominal':
				encode_attribute = True

			if encode_attribute:
				self._nominal_columns.append(attribute)

				attr_labelEncoder = LabelEncoder()
				attribute_values = attr_labelEncoder.fit_transform(attribute_values)

				self.columns_categories_[attribute] = [attr.decode() for attr in attr_labelEncoder.classes_.tolist()]
				del attr_labelEncoder

			data.T[index] = attribute_values
		
		del data_records

		self.data, self.target = data, target
		self.n_samples, self.n_features = self.data.shape


	def dummy_coding(self, nominal_columns='infer', drop_first=False):
		"""Dummy coding (One-Hot Encoding) of nominal categorical columns (features)
		
		Parameters:
			nominal_columns (:obj:`list`, int, str, 'all', default='infer'): List (str or int if singleton) of column 'names' (or absolute 'indices', if no column names) of nominal categorical columns to dummy code. ``nominal_columns='infer'`` autodetects nominal categorical columns. ``nominal_columns='all'`` implies all columns are nominal categorical. ``nominal_columns=None`` implies no nominal categorical columns.
			drop_first (bool, default=False): Whether to get k-1 dummies out of k categorical levels by removing the first level.
		
		Note:
			``nominal_columns`` parameter uses absolute column 'names' (or absolute column 'indices' if no names) as presented in the original data file.
		
		See also:
			`What is One Hot Encoding? Why And When do you have to use it? (Source: HackerNoon) <https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f>`_
				
		Examples:
			Illustration of **Dummy-Coding** of Nominal Categorical Columns

			::

			>>> from craved import eda
			>>> main = eda.eda()
			
			>>> from io import StringIO
			
			>>> data = '''	
			...	% Dataset: Automobiles (Source: UCI ML Repository)
			...	% Attributes :	symboling (ordinal) {-3, -2, -1, 0, 1, 2, 3} 
			...	%				body-style (nominal) {hardtop, wagon, sedan, hatchback, convertible}
			...	%				engine-size (continous) [61, 326]
			...	%				engine-location (nominal) {front, rear}
			...	% Target Attribute : symboling
			...
			...	3,convertible,130,front
			...	1,hatchback,152,front
			...	2,sedan,109,front
			...	3,hardtop,194,rear
			...	0,wagon,132,front
			...	-2,sedan,141,front
			...	3,convertible,194,rear
			...	-1,hatchback,122,front
			...	2,hardtop,97,front
			...	0,wagon,108,front
			...	'''
			
			>>> # Ignore lines starting with '%' as comment, i.e., comment='%'.
			...	# Use column 0 (attribute 'symboling') as target values to predict, i.e., target_col=0.
			...	# Encode nominal columns 1 and 3 (body-style and engine-location), i.e., categorical_cols=[1,3]
			... main.read_data_csv(StringIO(data), comment='%', target_col=0, encode_target=False, categorical_cols=[1,3])
			
			>>> # Print the processed data samples.
			...	print(main.data)
			[[   0.  130.    0.]
			 [   2.  152.    0.]
			 [   3.  109.    0.]
			 [   1.  194.    1.]
			 [   4.  132.    0.]
			 [   3.  141.    0.]
			 [   0.  194.    1.]
			 [   2.  122.    0.]
			 [   1.   97.    0.]
			 [   4.  108.    0.]]

			>>> # Print names (or absolute indices, if no names) of columns in data. 
			...	# Note: Column 0 was isolated as target values.
			...	print(main.columns_)
			Int64Index([1, 2, 3], dtype='int64')
			
			>>> # Print the names (or absolute indices, if no names) of nominal columns in data.
			... print(main._nominal_columns)
			[1, 3]

			>>> # Dummy code nominal columns inferred from data, i.e., nominal_columns='infer' (default).
			... main.dummy_coding()
			info: columns [1, 3] was/were infered as nominal column(s) for dummy coding

			>>> # Print the data samples post dummy-coding
			...	print(main.data)
			[[ 130.    1.    0.    0.    0.    0.    1.    0.]
			 [ 152.    0.    0.    1.    0.    0.    1.    0.]
			 [ 109.    0.    0.    0.    1.    0.    1.    0.]
			 [ 194.    0.    1.    0.    0.    0.    0.    1.]
			 [ 132.    0.    0.    0.    0.    1.    1.    0.]
			 [ 141.    0.    0.    0.    1.    0.    1.    0.]
			 [ 194.    1.    0.    0.    0.    0.    0.    1.]
			 [ 122.    0.    0.    1.    0.    0.    1.    0.]
			 [  97.    0.    1.    0.    0.    0.    1.    0.]
			 [ 108.    0.    0.    0.    0.    1.    1.    0.]]

			>>> # Print names of columns in data post dummy-coding.
			...	# Note: Dummy/indicator columns assume names of the form **'<original column name>_<nominal category binarized>'**
			...	print(main.columns_)
			Index([2, '1_0.0', '1_1.0', '1_2.0', '1_3.0', '1_4.0', '3_0.0', '3_1.0'], dtype='object')
		"""
		try:
			dataframe = pd.DataFrame(self.data, columns=self.columns_, dtype=np.number)

		except ValueError:
			print("warning: Data contains non-numeric features")
			dataframe = pd.DataFrame(self.data, columns=self.columns_)

		#if not (nominal_columns==[] or nominal_columns is None): # Both [] (empty list) and ``None`` are False Expressions
		if nominal_columns: # Evaluates to True if (nominal_columns!=[] and nominal_columns is not None)

			if isinstance(nominal_columns, str) and nominal_columns.casefold()=='infer':
				
				if hasattr(self, '_nominal_columns'):
					nominal_columns = self._nominal_columns if self._nominal_columns is not None else []
					print("info: columns {0} was/were infered as nominal column(s) for dummy coding".format(nominal_columns))

				else:
					print("error: could not infer nominal type columns from data")
					sys.exit(1)

			elif isinstance(nominal_columns, str) and nominal_columns.casefold()=='all':
				nominal_columns = self.columns_.copy()

			elif isinstance(nominal_columns, list) or isinstance(nominal_columns, str) or isinstance(nominal_columns, int):

				if isinstance(nominal_columns, str) or isinstance(nominal_columns, int):
					nominal_columns = [nominal_columns]

				if not set(nominal_columns).issubset(self.columns_):
					print("warning: Unknown columns names: {0} in argument to parameter 'nominal_columns' have been ignored".format( set(nominal_columns).difference(self.columns_) ))
					nominal_columns = list( set(nominal_columns).intersection(self.columns_) )

			else:
				print("error: Invalid arguments to parameter 'nominal_columns'. Accepted Arguments: {list of names of nominal columns, 'infer', 'all', None}")
				sys.exit(1)

			dataframe_dummy_coded = pd.get_dummies(dataframe, columns=nominal_columns, drop_first=drop_first)
			del dataframe

			self.data = dataframe_dummy_coded.values
			self.columns_ = dataframe_dummy_coded.columns

			del dataframe_dummy_coded
			del self._nominal_columns

			self.n_samples, self.n_features = self.data.shape

		else:
			print("info: No columns to dummy code (nominal_columns = {0})".format(nominal_columns.__repr__()))

	def standardize_data(self):
		"""Feature Scaling through Standardisation (or Z-score normalisation)

		See also:
			`Importance of Feature Scaling <http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html>`_
		"""
		if not hasattr(self, 'standard_scaler'):

			try:
				self.data = self.data.astype(np.float, copy=False)

			except ValueError:	
				print("error: Standardization of data failed due to presence of non-numeric features")
				sys.exit(1)

			self.standard_scaler = StandardScaler(copy=False)
			self.data = self.standard_scaler.fit_transform(self.data)

		else:
			print("info: Data already in Standard Normal Form")
		

	def destandardize_data(self):
		"""Scale back and shift features to original representation (i.e., as prior to Standardization)
		
		Note:
			Data should not have been modified post standardization for de-standardisation to return accurate original representation.
		"""
		if hasattr(self, 'standard_scaler'):
			self.data = self.standard_scaler.inverse_transform(self.data)
			del self.standard_scaler


	# TODO: Variable sampling sizes across samples.
	def random_sampling(self, bag_name, sample_size, n_iterations=10, file_prefix=None, location=__warehouse__+"BAGS/" if __warehouse__ is not None else None):
		"""Repeated Random Sampling of data with 'replacement across samples'

		Parameters:
			bag_name (str): Name of (to be created) folder that acts as a container for the sampled data bags.

			sample_size (int, float): Number of data samples in every bag. { ``int`` (range: 1 to n_samples):Absolute number of samples per bag, ``float`` (range: (0, 1] ):Number of samples per bag represented as a fraction of the total number of samples}
			n_iterations (int, default=10): Number of bags to be formed.
			
			file_prefix (str, default=None): Prefix for bag filenames. Bag filenames are of the form '<file_prefix>_bag<bag number>.p'.
			location (str, default='<path to Buddi-CRAVeD warehouse>/BAGS/'): Location to dump the sampled data bags. Defaults to the `BAGS` folder in Buddi-CRAVeD warehouse directory.

		Note:
			* Each sampled data bag file is an pickled dictionary of 'data' and 'target' attributes.
			* Each bag folder contains a file 'metadata.p' which is a pickled dictionary of metadata information about the original dataset (bagging timestamp, class distribution, n_samples, n_features, columns (features) information). 
			* The metadata 'timestamp' attribute (time of bagging in seconds since the Epoch as a float) can uniquely identify bags (in most cases).

		Examples:
			Illustration of 'Random Sampling' of datasets

			>>> from craved import eda
			>>> main = eda.eda()
			
			>>> from io import StringIO

			>>> # Dataset: WEATHER (source: Weka)
			...	data =	'''
			...	@relation WEATHER
			...
			...	@attribute outlook {sunny, overcast, rainy}
			...	@attribute temperature real
			...	@attribute humidity real
			...	@attribute windy {TRUE, FALSE}
			...	@attribute play {yes, no}
			...
			...	@data
			...	sunny,85,85,FALSE,no
			...	sunny,80,90,TRUE,no
			...	overcast,83,86,FALSE,yes
			...	rainy,70,96,FALSE,yes
			...	rainy,68,80,FALSE,yes
			...	rainy,65,70,TRUE,no
			...	overcast,64,65,TRUE,yes
			...	sunny,72,95,FALSE,no
			...	sunny,69,70,FALSE,yes
			...	rainy,75,80,FALSE,yes
			...	sunny,75,70,TRUE,yes
			...	overcast,72,90,TRUE,yes
			...	overcast,81,75,FALSE,yes
			...	rainy,71,91,TRUE,no
			...	'''
				
			>>> # The attribute 'play' constitutes target class values, i.e., target_attr='play'
			...	# Doesn't doesn't have numerical nominal attributes, i.e., num_categorical_attrs=None (default)
			... main.read_data_arff(StringIO(data), target_attr='play')

			>>> # Dummy code all nominal columns 'inferred' from data, i.e., nominal_columns='infer' (default)
			... main.dummy_coding()
			info: columns ['outlook', 'windy'] was/were infered as nominal column(s) for dummy coding

			>>> # Standardize of features in the dataset
			...	main.standardize_data()

			>>> # Draw 2 bags each of 10 samples (randomly sampled with replacement across bags) from the dataset
			...	'''	Dump the sampled bags in (a to be created)  folder 'WEATHER' in the location '<path to Buddi-CRAVeD warehouse>/BAGS/',
			...	i.e., bag_name='WEATHER', location=__warehouse__+'BAGS/' (default)	'''
			...	# Assumption: Buddi-CRAVeD warehouse has been setup (by running script :program:`craved-warehouse`)
			...	main.random_sampling(bag_name='WEATHER', sample_size=10, n_iterations=2)
			
			>>> import os

			# Change current directory to folder 'WEATHER' in the location '<path to Buddi-CRAVeD warehouse>/BAGS/'
			... os.chdir(eda.__warehouse__+'BAGS/WEATHER')
			
			>>> # Check the location for the sampled bags and associated metadata files
			...	os.listdir()
			['bag2.p', 'bag1.p', 'metadata.p']

			>>> # Load bag 'bag1.p' (a pickled dictionary with 'data' and 'target' attributes) into :obj:`eda` object
			... import pickle 
			>>> main_bag1 = eda.eda()
			>>> dataset = pickle.load( open('bag1.p','rb') )
			>>> data, target = dataset['data'], dataset['target']
			>>> main_bag1.load_data(data, target=target)
			
			>>> Print the data samples in bag 'bag1.p'
			... print(main_bag1.data)
			[[ 1.48889015  0.43962323  1.58113883 -0.74535599 -0.74535599  0.8660254 -0.8660254 ]
 			 [-0.87979873 -0.16575958 -0.63245553  1.34164079 -0.74535599  0.8660254 -0.8660254 ]
 			 [-0.24814836  1.34769743 -0.63245553 -0.74535599  1.34164079  0.8660254 -0.8660254 ]
 			 [-0.56397354  1.44859457 -0.63245553  1.34164079 -0.74535599  0.8660254 -0.8660254 ]
 			 [-1.5114491  -1.67921659  1.58113883 -0.74535599 -0.74535599 -1.15470054 1.15470054]
 			 [-0.72188614 -1.17473092 -0.63245553 -0.74535599  1.34164079  0.8660254 -0.8660254 ]
 			 [-0.24814836  0.84321176  1.58113883 -0.74535599 -0.74535599 -1.15470054 1.15470054]
 			 [ 1.80471534  0.33872609 -0.63245553 -0.74535599  1.34164079  0.8660254 -0.8660254 ]
 			 [ 0.22558942 -0.16575958 -0.63245553  1.34164079 -0.74535599  0.8660254 -0.8660254 ]
 			 [ 1.01515238  0.84321176 -0.63245553 -0.74535599  1.34164079 -1.15470054 1.15470054]]
			
			>>> # Print the target values of data samples in bag 'bag1.p'
			...	print(main_bag1.target)
 			[1 1 0 1 1 1 1 0 1 0]

 			>>> # Print metadata for original dataset and sampled bags (from file :file:`metadata.p`)
 			... metadata = pickle.load( open('metadata.p','rb') )
 			>>> for key, value in metadata.items():
 			...		print(key, ':', value)
			
			# No. of samples in original dataset
			n_samples : 14

			# No. of features in original preprocessed dataset.
			# Note: Number of features in original dataset increased from 4 to 7 due to dummy coding of nominal columns
 			n_features : 7				
						
			# Class-wise (aka. target value based) distribution of data samples in dataset
			classes : {0: 5, 1: 9}
			
			# Nominal categories (if any) in columns of original dataset (Note: Non-nominal (numeric) features assume value ``None``)
			column_categories : {'temperature': None, 'outlook': ['overcast', 'rainy', 'sunny'], 'windy': ['FALSE', 'TRUE'], 'humidity': None}
			
			# Sampling timestamp (in seconds since the Epoch).
			''' To print formatted timestamp output::
				
				>>> import time
				>>> print( time.asctime(time.localtime(metadata['timestamp'])) )
				Wed Jan 10 12:38:27 2018
			'''
			timestamp : 1515568107.551725
		
			# Names of columns in dataset post preprocessing(column value encoding and dummy coding of nominal columns)
			column_names : Index(['temperature', 'humidity', 'outlook_0.0', 'outlook_1.0', 'outlook_2.0',
       			'windy_0.0', 'windy_1.0'],
      			dtype='object')
		"""

		# Resolving LOCATION for bagged samples
		if location is not None:
			
			location = os.path.abspath(os.path.expanduser(location))

			try:
				os.chdir(location)

			except FileNotFoundError:	
				print("error: Failed to resolve location '%s'"%location)
				sys.exit(1)

		else:
			print("error: Buddi-CRAVeD 'warehouse' not setup. Specify an user path for sampled data bags.")
			sys.exit(1)

		try:
			os.mkdir(bag_name)
			os.chdir(bag_name)

		except OSError as err:
			print("error: Unable to write sampled data bags to disk.\n{0}".format(err))
			sys.exit(1)

		# Resolving SIZE of bagged samples
		if isinstance(sample_size, int) and (sample_size>0 and sample_size<=self.n_samples):
			pass

		elif isinstance(sample_size, float) and (sample_size>0.0 and sample_size<=1.0):
			sample_size = ceil(sample_size*self.n_samples)

		else:
			print("error: Invalid sampling size encountered")
			sys.exit(1)

		# Resolving FILE PREFIX for bagged samples
		if file_prefix is None:
			file_prefix = ''

		else:
			file_prefix = file_prefix + '_'

		# Repeated Sampling of data
		for iteration in range(n_iterations):

			sampled_data = dict.fromkeys(['data', 'target'])

			choosen_indices = np.random.choice(np.arange(self.n_samples),size=sample_size,replace=False)
			sampled_data['data'], sampled_data['target'] = self.data[choosen_indices], self.target[choosen_indices] if self.target is not None else None

			pickle.dump(sampled_data, open(file_prefix + "bag"+str(iteration+1)+".p","xb"))

			del sampled_data

		# Metadata of data
		from time import time
		metadata = 	{
						'timestamp':time(), # Uniquely identifies baggings (with probability ~= 1)
						'classes':label_cnt_dict(self.target) if self.target is not None else None,
						
						'n_samples':self.n_samples, # Not inferrable from classes, if target=None
						'n_features':self.n_features,

						'column_names':self.columns_,
						'column_categories':self.columns_categories_ if hasattr(self, 'columns_categories_') else None
					}

		pickle.dump(metadata, open("metadata.p", "xb"))

	def stratified_sampling(self, bag_name, sample_size, n_iterations=10, file_prefix=None, location=__warehouse__+"BAGS/" if __warehouse__ is not None else None):

		# Resolving LOCATION for bagged samples
		if location is not None:
			
			location = os.path.abspath(os.path.expanduser(location))

			try:
				os.chdir(location)

			except FileNotFoundError:	
				print("error: Failed to resolve location '%s'"%location)
				sys.exit(1)

		else:
			print("error: Buddi-CRAVeD 'warehouse' not setup. Specify an user path for sampled data bags.")
			sys.exit(1)

		try:
			os.mkdir(bag_name)
			os.chdir(bag_name)

		except OSError as err:
			print("error: Unable to write sampled data bags to disk.\n{0}".format(err))
			sys.exit(1)


		# Validating the sample size parameter
		if isinstance(sample_size, int) and (sample_size>0 and sample_size<=self.n_samples):
			pass

		elif isinstance(sample_size, float) and (sample_size>0.0 and sample_size<=1.0):
			pass

		else:
			print("error: Invalid sampling size encountered")
			sys.exit(1)
		

		if file_prefix is None:
			file_prefix = ''

		else:
			file_prefix = file_prefix + '_'


		if self.target is not None:

			sss = StratifiedShuffleSplit(n_splits=n_iterations, test_size=sample_size)
			sss.get_n_splits(self.data, self.target)
			bag_number = 0

			for train_index, test_index in sss.split(self.data, self.target):

				sampled_data = dict.fromkeys(['data', 'target'])

				sampled_data['data'], sampled_data['target'] = self.data[test_index], self.target[test_index]
				pickle.dump(sampled_data, open(file_prefix + "bag"+str(bag_number+1)+".p","xb"))
				bag_number = bag_number + 1
				del sampled_data

			# Metadata of data
			from time import time
			metadata = 	{
							'timestamp':time(), # Uniquely identifies baggings (with probability ~= 1)
							'classes':label_cnt_dict(self.target) if self.target is not None else None,
							
							'n_samples':self.n_samples, # Not inferrable from classes, if target=None
							'n_features':self.n_features,

							'column_names':self.columns_,
							'column_categories':self.columns_categories_ if hasattr(self, 'columns_categories_') else None
						}

			pickle.dump(metadata, open("metadata.p", "xb"))

		else:
			print("error: stratified sampling failed due to absence of target attribute")
			sys.exit(1)




	def perform_kmeans(self, n_clusters='gap', **kargs):
		"""Perform K-Means Clustering on the data

		n_clusters ({int, 'gap', 'n_classes'}, default='gap'): number (``int``) of clusters in the data. ``gap`` implies estimate the optimal number of clusters from data using Gap Statistics. ``n_classes`` implies uses number of classes in data as number of clusters.
		**kargs: Other Keyword arguments (parameters) accepted by object :`sklearn.cluster.KMeans` constructor (Keyword Arguments: n_init, max_iter, verbose, n_jobs).
		
		Note:
			'Gap Statistics' may produce inconsistent K (optimal number of cluster) estimation across trials for datasets without clear distinct isotropic gaussian blob clusters.
		
		See also:
			* The method :func:`craved.eda.perform_kmeans` is built upon `scikit-learn's KMeans Clustering API`_ (:obj:`sklearn.cluster.KMeans`).
			* The Data Science Lab : `Finding the K in K-Means Clustering`_  (Gap Statistics)

		Examples:
			Illustration of performing KMeans Clustering on synthetic dataset::

			>>> from craved import eda
			>>> main = eda.eda()

			>>> # Generate synthetic dataset (with istropic gaussian blobs clusters) using :func:`sklearn.datasets.make_blobs`
			... from sklearn.datasets import make_blobs
			>>> data, target = make_blobs(n_samples=100, n_features=2, centers=3)

			>>> # Load the synthetic dataset into the eda object :obj:`main`
			...	main.load_data(data, target)

			>>> # Perform K-Means Clustering on the data
			... main.perform_kmeans(n_clusters='gap')
			info: Data implicilty Standardized (aka Z-Score Normalised) for K-Means Clustering
			< ... some gap statistics (optimal n_clusters estimation) info goes here ... >
			info: Optimal number of clusters in data, K=3 (as inferred using gap statistics)

			inertia : 8.030120482
			clusters : {0: 33, 1: 34, 2: 33}
			parameters : {'verbose': 0, 'precompute_distances': 'auto', 'init': 'k-means++', 'tol': 0.0001, 'n_jobs': 1, 'random_state': None, 'max_iter': 300, 'n_init': 10, 'algorithm': 'auto', 'copy_x': True, 'n_clusters': 3}
			n_clusters : 3
			cluster_centers : [[ 0.54512904 -1.38171852]
			 [-1.36053651  0.7996122 ]
			 [ 0.85663585  0.55787564]]
			labels : [1 0 0 2 1 2 0 1 1 2 2 2 1 1 1 0 0 1 2 0 0 0 1 0 2 2 1 1 2 2 1 0 1 0 2 0 0
			 0 2 1 2 1 0 1 0 0 0 0 1 1 1 0 0 2 0 1 0 2 1 2 1 2 2 1 0 2 1 2 2 1 2 0 1 1
			 2 0 0 2 0 2 1 0 0 2 2 2 0 0 1 1 2 2 1 1 0 1 2 1 2 2]

			.. _`Finding the K in K-Means Clustering`: https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
			 .. _`Scikit-Learn's KMeans Clustering API`: scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
		"""
		
		print("info: Performing KMeans Clustering")

		if not hasattr(self, 'standard_scaler'):
			self.standardize_data()
			print("info: Data implicilty Standardized (aka Z-Score Normalised) for K-Means Clustering")

		if isinstance(n_clusters, int):
			pass

		# Estimate optimal 'number of clusters' (k) from data using gap statistics
		elif isinstance(n_clusters, str) and n_clusters.casefold()=='gap':

			kmin = 1
			kmax = min(10, self.n_samples-1)
			bestKValue = -1

			# TODO: Handle boundary cases
			while bestKValue < 0 and kmax>kmin:
				print("info: searching for optimal number of clusters in range %d to %d"%(kmin, kmax))
				bestKValue = gap_statistics(self.data, kmax=kmax, kmin=kmin, cluster_algorithm=KMeans, **kargs)
				
				kmin = kmax+1
				kmax += min(10, (self.n_samples-1) - kmax)

			if bestKValue < 0:
				print("error: Gap statistics couldn't estimate the optimal number of clusters in data")
				sys.exit(1)

			n_clusters = bestKValue
			print("info: Optimal number of clusters in data, K=%d (as inferred using gap statistics)"%n_clusters)
			if(n_clusters < 2 ):
				print("info:gap statistics returned optimal clusters =1 ,FORCING n_clusters = 2 ")
				n_clusters = 2
			

		# 'number of clusters' to find = 'number of classes' in the labelled dataset
		elif isinstance(n_clusters, str) and n_clusters.casefold()=='n_classes':
			if self.target is not None:
				if hasattr(self, 'classes_') and self.classes_ is not None:
					n_clusters, = self.classes_.shape
				else:
					n_clusters, = np.unique(self.target).shape

				print("info: number of clusters in data, K=%d (equal to number of classes)"%n_clusters)

			else:
				print("error: number of classes in data couldn't be determined due to absence of target class info.")
				sys.exit(1)
		
		else:
			print("error: invalid argument for parameter 'n_clusters'. Accepted arguments: {int, 'gap', 'n_classes'}")
			sys.exit(1)

		kmeans_clusterer = KMeans(n_clusters=n_clusters, **kargs)
		kmeans_clusterer.fit(self.data)
		
		self.kmeans_results = {
								"parameters":kmeans_clusterer.get_params(),
								"labels":kmeans_clusterer.labels_,
								"n_clusters":n_clusters,
								'clusters':label_cnt_dict(kmeans_clusterer.labels_),
								"cluster_centers":kmeans_clusterer.cluster_centers_,
								"inertia":kmeans_clusterer.inertia_
							}

		print_dict(self.kmeans_results)


	def perform_hierarchial(self, n_clusters='gap', **kargs):
		"""Perform Ward's Hierarchial Clustering on the data

		n_clusters ({int, 'gap', 'n_classes'}, default='gap'): number (``int``) of clusters in the data. ``gap`` implies estimate the optimal number of clusters from data using Gap Statistics. ``n_classes`` implies uses number of classes in data as number of clusters.
		**kargs: Other Keyword arguments (parameters) accepted by object :`sklearn.cluster.AgglomerativeClustering` constructor (Keyword Arguments: affinity, linkage, memory).
		
		Note:
			'Gap Statistics' may produce inconsistent K (optimal number of cluster) estimation across trials for datasets without clear distinct clusters.
		
		See also:
			* The method :func:`craved.eda.perform_hierarchial` is built upon `scikit-learn's Agglomerative Clustering API`_ (:obj:`sklearn.cluster.AgglomerativeClustering`).
			* Research Paper : `Estimating the number of clusters in a data set via the gap statistics`_ 

		Examples:
			Illustration of performing Ward's Agglomerative Hierarchial Clustering on synthetic dataset::

			>>> from craved import eda
			>>> main = eda.eda()

			>>> # Generate synthetic dataset (with istropic gaussian blobs clusters) using :func:`sklearn.datasets.make_blobs`
			... from sklearn.datasets import make_blobs
			>>> data, target = make_blobs(n_samples=100, n_features=2, centers=3)

			>>> # Load the synthetic dataset into the eda object :obj:`main`
			...	main.load_data(data, target)

			>>> # Perform Agglomerative Hierarchial Clustering on the data
			... main.perform_hierarchial(n_clusters='gap')
			info: Data implicilty Standardized (aka Z-Score Normalised) for Hierarchial Clustering
			< ... some gap statistics (optimal n_clusters estimation) info goes here ... >
			info: Optimal number of clusters in data, K=3 (as inferred using gap statistics)
			
			n_clusters : 3
			labels : [1 2 2 1 0 2 1 2 1 2 2 2 1 1 2 1 0 2 1 0 1 0 2 0 2 0 2 1 1 2 1 1 2 2 1 1 0
			 0 2 0 0 0 0 1 0 0 2 2 2 1 1 0 1 0 1 2 1 2 1 2 0 1 0 0 0 2 2 2 0 0 0 1 1 1
			 0 1 0 0 2 2 0 0 2 1 1 1 2 2 1 0 2 0 1 0 0 1 0 0 1 2]
			clusters : {0: 34, 1: 34, 2: 32}
			parameters : {'affinity': 'euclidean', 'connectivity': None, 'pooling_func': <function mean at 0x7f991ff63268>, 'n_clusters': 3, 'memory': None, 'compute_full_tree': 'auto', 'linkage': 'ward'}

		.. _`scikit-learn's Agglomerative Clustering API`: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
		.. _`Estimating the number of clusters in a data set via the gap statistics`: http://web.stanford.edu/~hastie/Papers/gap.pdf
		"""
		
		print("info: Performing Hierarchial Clustering")

		if not hasattr(self, 'standard_scaler'):
			self.standardize_data()
			print("info: Data implicilty Standardized (aka Z-Score Normalised) for Hierarchial Clustering")

		if isinstance(n_clusters, int):
			pass

		# Estimate optimal 'number of clusters' (k) from data using gap statistics
		elif isinstance(n_clusters, str) and n_clusters.casefold()=='gap':

			kmin = 1
			kmax = min(10, self.n_samples-1)
			bestKValue = -1

			# TODO: Handle boundary cases
			while bestKValue < 0 and kmax>kmin:
				print("info: searching for optimal number of clusters in range %d to %d"%(kmin, kmax))
				bestKValue = gap_statistics(self.data, kmax=kmax, kmin=kmin, cluster_algorithm=AgglomerativeClustering, **kargs)
				
				kmin = kmax+1
				kmax += min(10, (self.n_samples-1) - kmax)

			if bestKValue < 0:
				print("error: Gap statistics couldn't estimate the optimal number of clusters in data")
				sys.exit(1)

			n_clusters = bestKValue
			print("info: Optimal number of clusters in data, K=%d (as inferred using gap statistics)"%n_clusters)
			if(n_clusters < 2 ):
				print("info:gap statistics returned optimal clusters =1 ,FORCING n_clusters = 2 ")
				n_clusters = 2
			

		# 'number of clusters' to find = 'number of classes' in the labelled dataset
		elif isinstance(n_clusters, str) and n_clusters.casefold()=='n_classes':
			if self.target is not None:
				if hasattr(self, 'classes_') and self.classes_ is not None:
					n_clusters, = self.classes_.shape
				else:
					n_clusters, = np.unique(self.target).shape

				print("info: number of clusters in data, K=%d (equal to number of classes)"%n_clusters)
				
			else:
				print("error: number of classes in data couldn't be determined due to absence of target class info.")
				sys.exit(1)
		
		else:
			print("error: invalid argument for parameter 'n_clusters'. Accepted arguments: {int, 'gap', 'n_classes'}")
			sys.exit(1)

		if self._hdf5_file is not None:
			try:
				hierarchial_clusterer = AgglomerativeClustering(n_clusters=n_clusters, hdf5_file=self._hdf5_file, **kargs)

			except TypeError:
				print("warning: Accompanying package 'sklearn-large' not installed. Cannot write intermediate data structures to HDF5 file.")
				hierarchial_clusterer = AgglomerativeClustering(n_clusters=n_clusters, **kargs)

		else:
			hierarchial_clusterer = AgglomerativeClustering(n_clusters=n_clusters, **kargs)
		
		try:
			hierarchial_clusterer.fit(self.data)

		except MemoryError:
			print("error: Data too large to be processed on this machine. Install accompanying package 'sklearn-large' (if you haven't) and try again ")
			sys.exit(1)
		
		self.hierarchial_results = {
										"parameters":hierarchial_clusterer.get_params(),
										"labels":hierarchial_clusterer.labels_,
										"n_clusters":n_clusters,
										'clusters':label_cnt_dict(hierarchial_clusterer.labels_)
									}

		print_dict(self.hierarchial_results)


def label_cnt_dict(labels):
	unique, counts = np.unique(labels, return_counts=True)
	return dict(zip(unique, counts))

def print_dict(dictionary):
	for key,value in dictionary.items():
		print(key,value,sep=" : ")

def visualise_2D(x_values,y_values,labels=None,class_names=None):
	"""Visualise clusters of selected 2 features"""

	sns.set_style('white')
	sns.set_context('poster')
	sns.set_color_codes()

	plot_kwds = {'alpha' : 0.5, 's' : 50, 'linewidths':0}

	frame = plt.gca()
	frame.axes.get_xaxis().set_visible(False)
	frame.axes.get_yaxis().set_visible(False)

	if labels is None:
		plt.scatter(x_values,y_values,c='b',**plot_kwds)

	else:
		pallete=sns.color_palette('dark',np.unique(labels).max()+1)
		colors=[pallete[x] if x>=0 else (0.0,0.0,0.0) for x in labels]
		plt.scatter(x_values,y_values,c=colors,**plot_kwds)
		legend_entries = [mpatches.Circle((0,0),1,color=x,alpha=0.5) for x in pallete]

		if class_names is None:
			legend_labels = range(len(pallete))

		else:
			legend_labels = ["class "+str(label)+" ( "+str(name)+" )" for label,name in enumerate(class_names)]

		plt.legend(legend_entries,legend_labels,loc='best')
		
	plt.show()


def visualise_3D(x_values,y_values,z_values,labels=None):
	"""Visualise clusters of selected 3 features -- plotly"""
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')

	plot_kwds = {'alpha' : 0.5, 's' : 50, 'linewidths':0}

	if labels is None:
		ax.scatter(x_values,y_values,z_values,c='b',**plot_kwds)

	else:
		pallete=sns.color_palette('dark',np.unique(labels).max()+1)
		colors=[pallete[x] if x>=0 else (0.0,0.0,0.0) for x in labels]
		ax.scatter(x_values,y_values,z_values,c=colors,**plot_kwds)

	plt.show()

def euclidean_distance(vector1,vector2,squared=False):
	"""calculates euclidean distance between two vectors
	Keyword arguments:
	vector1 -- first data point (type: numpy array)
	vector2 -- second data point (type: numpy array)
	squared -- return square of euclidean distance (default: False)
	"""
	euclidean_distance=np.sum((vector1-vector2)**2,dtype=np.float64)
	if squared is False:
		euclidean_distance=np.sqrt(euclidean_distance,dtype=np.float64)
	return euclidean_distance

#Flatten complex 'multi-dimensional' list or ``np.ndarray``s
def flatten_list(data):

	if isinstance(data, int) or isinstance(data, float):
		return list([data])

	if isinstance(data, np.ndarray):
		data = data.tolist()

	flattened_list = []
	for element in data:
		flattened_list = flattened_list + flatten_list(element)

	return flattened_list

# max number of classes in a nominal variables for dataset with ``n_samples`` data points
def max_classes_nominal(n_samples):
	
	# Result of quadratic regression on "n_samples" -> "max classes in nominal columns"
	reg_coefs = np.array([  8.54480458e-03,   1.31494511e-08])
	reg_intercept = 14.017948334463796
	
	if n_samples <= 16:
		return ceil(n_samples/3)
	
	elif n_samples <= 100000:
		return ceil( min(np.sum([n_samples, n_samples*n_samples]*reg_coefs) + reg_intercept, n_samples/4) )
	
	else:
		return  n_samples/100

def gap_statistics(data, kmax=10, kmin=1, B=10, cluster_algorithm=KMeans, **kargs):
		"""Estimating the number of clusters in a data set via the gap statistics method

		Parameters:
			data (:obj:`np.ndarray`): the data samples to cluster
			kmax (int, default=10): the start value of the range of 'K' values to search for the optimal number of clusters
			kmin (int, default=1): the stop value of the range of 'K' values to search for the optimal number of clusters
			B (int, default=10): number of reference datasets (uniformly sampled from the box region of original data) used to generate **standard** K v/s log(wk) graph
			cluster_algorithm (callable, default: :func:`sklearn.cluster.KMeans`): a sklearn (aka scikit-learn) complient 'clusterer' object constructor which accepts ``n_clusters`` (number of clusters) as a parameter.
			**kargs: other parameters accepted by 'clusterer' object constructor
		
		Returns:
			Optimal number of clusters (k) in data, if k in the range from kmin to kmax. Returns -1 otherwise.

		Note:
			* The estimated value of 'k' (optimal number of cluster) is prone to change over runs (for data with no distinct blob clusters) due to random reference samples generated by the algorithm.
			* Data must be standardized (aka Z-Score Normalised) prior to 'k' (optimal number of clusters) value estimation using gap statistics.

		See also:
			* The Data Science Lab : `Finding the K in K-Means Clustering`_ 
			* Research Paper : `Estimating the number of clusters in a data set via the gap statistics`_
		
		Examples:
			Illustration of **computing the optimal number of clusters** in data using GAP STATISTICS::
			
			>>> from craved import eda
			
			>>> # Generate toy datasets (with K=7 isotropic Gaussian blobs) using :func:`sklearn.datasets.make_blobs`
			... from sklearn.datasets import make_blobs
			>>> data, target = make_blobs(n_samples=1000, n_features=3, centers=7)
			
			>>> # Load data into an eda object
			... main = eda.eda()
			>>> main.load_data(data, target)

			>>> # Standardise (Z-Score Normalisation) of features
			... main.standardize_data()	
			
			>>> # Compute best K value in the range from 2 to 10, i.e, K=range(2,11) (default)
			... bestKValue = eda.gap_statistics(main.data)
			>>> print("Optimal Number of clusters = %d"%bestKValue)
			Optimal Number of clusters = 7
		
		.. _`Finding the K in K-Means Clustering`: https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
		.. _`Estimating the number of clusters in a data set via the gap statistics`: http://web.stanford.edu/~hastie/Papers/gap.pdf
		"""


		bestKValue = gap.gap_statistics(data, kmax=kmax, kmin=kmin, B=B, cluster_algorithm=cluster_algorithm, **kargs)
		return bestKValue
