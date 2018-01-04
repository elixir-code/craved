__warehouse__ = None

#Check if craved warehouse has been setup
try:
	#chek for ambiguity in package __path__
	if len(__path__) == 1:
		craved_dir_path = __path__[0]

	with open(craved_dir_path + "/craved_warehouse.dat","r") as config_file:
		__warehouse__ = config_file.read().strip()

#catch all exception here and ignore
except:
	pass

finally:
	from . import eda, internal_indices, external_indices, supervised
