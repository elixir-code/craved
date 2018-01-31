""" Script to perform repeated random sampling (with replacement across samplings) of datasets into bags 
Supported Data Formats: CSV, LIBSVM, ARFF """

from craved import eda
main = eda.eda()

# Path and file name of the data file
filename = '<replace with path and filename of CSV, LIBSVM or ARFF data file>'





# Uncomment the following code segment for CSV data file -- start
"""

''' column (aka. feature) seperator, i.e, sequence of characters (or a regex) that seperate two features of a data sample '''

seperator = ',' # default
# seperator = '\s+' # combination of spaces and tabs
# seperator = None # autodetect delimiter (warning:  may not work with certain combinations of parameters)
# seperator = '<replace with a character or regular expression>' # examples: '\r\t'


''' number of initial lines <int> or list of line numbers <list> of the data file to skip before start of data samples 
Note: include the commented and blank lines in the count of intial lines to skip. Don't skip over header line (with column names) if any '''

skiplines = None # default: no lines to skip (aka. skiplines = 0)
# skiplines = <replace with number of initial lines to skip>
# skiplines = [<replace with list of line numbers to skip>]


''' relative zero-index based line number of header line containing column names to use
Note: 
	* The indexing starts (from line number=0) for lines immediately following the skipped lines.
	* Blank lines are ignored in the indexing.
	* All lines following the skipped lines until the header line are ignored.
'''

headerline = None # default: No header lines (containing column names) to use.
# headerline = 0 # the first non-blank line following skipped lines contains column names to use.
# headerline = <replace with relative zero-index of header line>


''' List of columns to use from the data file 
Note: 
	* Use list of column names (as inferred from header line) or zero-based column indices (only if no header line)
	* Include the 'target' value column in the list of columns to use
'''

usecolumns = None # default: Use all columns from data file
# usecolumns = [<replace with list of columns to use>]


''' relative zero-based index (among selected columns) or column name (as inferred from header line) of 'target' values column '''

targetcol = -1 # default: last column among list of selected columns
# targetcol = 0 # first column among list of selected columns
# targetcol = '<replace with name of target column as inferred from header line (if any)>'


''' Should target values be treated as nominal values and be encoded (with zero-indexed integers) '''

encode_target = True # default: Encode target values
# encode_target = False # Target values represent a non-nominal (or continous) feature and should not be encoded


''' List of column names (as inferred from header line) or absolute column indices (index of column as in data file) if no header line
of nominal categorical columns to encode. 

Note:
	* nominal_cols='infer' infers all string dtype columns with relatively large number of unique entries as 'string' or 'date' features and drops them from the dataset.
'''

nominal_cols = 'infer' # default: infer all string dtype columns with reasonably few unique entries as nominal columns. See Note (1).
# nominal_cols = 'all' # All columns are nominal categorical
# nominal_cols = None # No nominal categorical columns
# nominal_cols = [<list of nominal categorical columns to encode>]


''' List of strings to be inferred as NA (Not Available) or NaN (Not a Number) in addition to the default NA string values.
Default NA values :  ‘’, ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’, ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘N/A’, ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’
'''

na_values = None # default: no additional strings (asides default strings) to infer as NA/NaN
# na_values = [<list of additional strings to infer as NA/NaN>]


''' Dictionary of other keyword arguments accepted by :func:`pandas.read_csv` (Keyword Arguments: comment, lineterminator, ...) '''
kargs = {}


main.read_data_csv(filename, sep=seperator, skiprows=skiplines, header_row=headerline, usecols=usecolumns, target_col=targetcol, encode_target=encode_target, categorical_cols=nominal_cols, na_values=na_values, nrows=None, **kargs)
"""
# Uncomment the above code segment for CSV data file -- end








# Uncomment the following code segment for LIBSVM data file -- start
"""
Note: The column indices are one-based (i.e., [1 ... n_features]) irrespective of actual indices in the data file

main.read_data_libsvm(filename)
"""
# Uncomment the above code segment for LIBSVM data file -- end








# Uncomment the following code segment for ARFF data file -- start
"""

''' Attribute name of the target column '''

target_attr = 'class' # default: 'class' attribute contains the target values
# target_attr = '<replace with attribute name of target column>'


''' Should target values be encoded (with zero-indexed integers) '''

encode_target = 'infer' # default: encode if target attribute is nominal type and doesn't otherwise (i.e., if numeric type)
# encode_target = True
# encode_target = False


''' List of 'names' of numeric attributes to be inferred as nominal and to be encoded.
Note: All nominal attributes are implicitly encoded and should not be included in the list. '''

numeric_nominal_attrs = None # default: No numeric attributes are to be inferred as nominal
# numeric_nominal_attrs = [<replace with list of numeric type attributes to be inferred as nominal and encoded>]


# Note: Data samples with any NA/NaN (i.e., '?') features are implicitly dropped

main.read_data_arff(filename, target_attr=target_attr, encode_target=encode_target, num_categorical_attrs=numeric_nominal_attrs)
"""
# Uncomment the above code segment for ARFF data file -- end








""" Perform dummy coding of nominal columns (or features) """

''' List of column names (as inferred from 'header line in CSV' or 'metadata section in ARFF') or 
column indices (if 'no header line' in CSV) of nominal columns to dummy code '''


nominal_columns = 'infer' # Default: Use list of nominal columns supplied to (or inferred by) :func:`read_data_csv` or :func:`read_data_arff`
# nominal_columns = 'all' # All columns are nominal columns and are to be dummy coded
# nominal_columns = None # No nominal columns to dummy code
# nominal_columns = [<list of nominal columns to dummy code>]

main.dummy_coding(nominal_columns=nominal_columns)





main.standardize_data()





""" Perform repeated random sampling (with replacement across samplings) of dataset into bags """

# Name of folder that contains sampled bags and metadata files
dataset_name = '<Name of the dataset>'

# No. of samples in each bag (aka. sampling)
sampling_size = 3000
# sampling_size = <replace with an integer denoting number of samples per bag>


# Number of samplings (aka. bagging) to perform
n_iterations = 10
# n_iterations = <replace with an integer denoting number of sampling to perform>

main.random_sampling(dataset_name, sampling_size, n_iterations=n_iterations)