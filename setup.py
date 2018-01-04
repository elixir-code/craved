from setuptools import setup, find_packages

#read the README.rst file for use as long description
def readme():
	with open('README.rst') as f:
		return f.read()

setup(
	name = "craved", #name of the package
	version = "0.1a1", #version 0.1 alpha version 1
	packages = find_packages(), # automatically find packeges -- ['gap','sample']

	classifiers = [
		'Development Status :: 2 - Pre-Alpha',
		'License :: OSI Approved :: BSD License',
		'Programming Language :: Python :: 3 :: Only',
		'Topic :: Scientific/Engineering',		
	],

	#Issue : numpy dependency for hdbscan built
	install_requires = [
		'matplotlib', 
		'seaborn', 
		'scipy', 
		'pandas', 
		'sklearn', 
		'h5py',
		'hdbscan', 
		'numpy', 
	],

	author = "R Mukesh",
	author_email = "reghu.mukesh@gmail.com",
	
	description = "Buddi-CRAVeD",
	long_description = readme(),
	
	license = "3-clause BSD",
	url = "https://github.com/elixir-code/craved",

	keywords = 'cluster analysis validation visualisation internal external indices',

	#script 'craved-home' sets up directory structure to store intermediate files
	scripts=['bin/craved-warehouse'],

	include_package_data=True,
)
