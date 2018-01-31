.. highlight:: console

.. _setup:

Setting up the Environment [#]_
===============================

The suite interfaces extensively with some external standard `Python3 <https://docs.python.org/3/>`_ APIs to accomplish its tasks. It is essential that we get these :download:`dependencies <documents/requirements.txt>` [#]_ up and running, before we can get started.

Installing the Dependencies
---------------------------

`Matplotlib <https://matplotlib.org/>`_ (a plotting library for Python) powers the data visulation capabilities of the suite. It is built upon `Tkinter <https://docs.python.org/3/library/tkinter.html>`_ (**Tk inter**\ face for Python). To install tkiner::

	$ sudo apt-get install python3-tk

It is **strongly recommended** to use :program:`pip` and :program:`virtualenv` for installing python packages specific to projects. 

|more| For instructions on **installing and setting up pip and virtualenv**, refer `Installing packages using pip and virtualenv <https://packaging.python.org/guides/installing-using-pip-and-virtualenv/>`_.

.. note:: Ensure that you **activate your virtualenv** (if you have created one) before installing python packages.

The suite is built upon `Numpy <http://www.numpy.org/>`_, the fundamental package for **numerical and scientific computing** in Python. To install **numpy**::

	(env) $ python3 -m pip install numpy

Installing Buddi-CRAVeD Suite
-----------------------------

Installing from PyPI
~~~~~~~~~~~~~~~~~~~~

The **Buddi-CRAVeD** (alias `craved <https://pypi.python.org/pypi/craved>`_) package is available on `PyPI <https://pypi.python.org/pypi>`_ **(the Python Package Index)**. :program:`Pip` allows for a **one-step installation** of the package and its dependencies using::

	(env) $ python3 -m pip install craved

Installing from Source
~~~~~~~~~~~~~~~~~~~~~~

Alternatively, the **Buddi-CRAVeD** suite (source code) can be also be download from `https://github.com/elixir-code/craved.git <https://github.com/elixir-code/craved.git>`_ . ::

	$ git clone https://github.com/elixir-code/craved.git

:program:`Pip` allows for a **one-step installation** of the package and its dependencies from **local source** using::

	(env) $ python3 -m pip install <path to source directory>

|more| For instructions on `installing packages from VCS <https://packaging.python.org/tutorials/installing-packages/#installing-from-vcs>`_ (Version Control System) or `installing packages from local source <https://packaging.python.org/tutorials/installing-packages/#installing-from-a-local-src-tree>`_, refer `Python Packaging User Guide <https://packaging.python.org/tutorials/installing-packages/>`_.

Preparing the Data Warehouse
----------------------------

The **Buddi-CRAVeD warehouse** directory functions as an aggregated store for intermediate data structures, sampled datasets and accumulated results.

To configure and setup, an **empty directory** ( of your choice ) as the suite's data warehouse, execute the ":program:`craved-warehouse`" script in the **terminal** from the chosen warehouse directory. ::

	$ craved-warehouse

.. note:: The **'craved-warehouse'** script, that configures and sets up the package's warehouse can be invoked on **empty directories only**.

.. warning::  The **craved** doesn't allow **multiple warehouses** to be configured simaltaneously. Successful reinvokations of the **'craved-warehouse'** script on other directories will force the previous configuration to become invalid.

Extending Support for Large Datasets
------------------------------------

The :ref:`Buddi-CRAVeD <Buddi-CRAVeD>` suite's enhanced support for **cluster analysis of "larger" datasets** is enabled through our modified versions of the companion libraries -- `scikit-learn <http://scikit-learn.org/>`_ and `scipy <https://www.scipy.org/>`_.

These libraries in part derive their numerical computation capabilities from `ATLAS <http://math-atlas.sourceforge.net/>`_ **(Automatically Tuned Linear Algebra Software)**. To install **ATLAS**::

	$ sudo apt-get install libatlas-base-dev

The python **wheel** formats (built for linux systems) of the modified companion libraries can be downloaded from `sourceforge <https://sourceforge.net/projects/craved-support/files/>`_ (project : **craved-support**) - `scikit_learn-0.18.1-cp35-cp35m-linux_x86_64.whl <https://sourceforge.net/projects/craved-support/files/scikit_learn-0.18.1-cp35-cp35m-linux_x86_64.whl/download>`_ and `scipy-0.19.1-cp35-cp35m-linux_x86_64.whl <https://sourceforge.net/projects/craved-support/files/scipy-0.19.1-cp35-cp35m-linux_x86_64.whl/download>`_.

:program:`Pip` allows for **easy overwrite and installation** of the **remote** wheels. 

::
	
	(env) $ python3 -m pip uninstall scikit-learn
	(env) $ python3 -m pip install --use-wheel --no-index --find-links=https://sourceforge.net/projects/craved-support/files/scikit_learn-0.18.1-cp35-cp35m-linux_x86_64.whl scikit-learn

::

	(env) $ python3 -m pip uninstall scipy
	(env) $ python3 -m pip install --use-wheel --no-index --find-links=https://sourceforge.net/projects/craved-support/files/scipy-0.19.1-cp35-cp35m-linux_x86_64.whl scipy

|more| For instructions on the `usage of pip and wheel utilities <https://wheel.readthedocs.io/en/stable/#usage>`_ for installing **remote and local wheels**, refer to the `Wheel documentation <https://wheel.readthedocs.io/en/stable/>`_.

.. rubric:: Footnotes

.. [#]	The instructions for setting up the environment are specific to :program:`Ubuntu` based operating systems. However, it can replicated for other **Linux** Distros and **Windows** Systems.

.. [#] 	The list of dependencies were generated on a python :program:`virtualenv` created exclusively for the project and using :program:`pip`

		::

		(env) $ python3 -m pip freeze

.. |more| image:: images/more-info.png
		  :align: middle
		  :alt: more info
