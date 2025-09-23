Usage Guide
===========

Installation
------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/epfl-radio-astro/tabascal.git

Install via pip (CPU-only):

.. code-block:: bash

   pip install -e ./tabascal/

Or with GPU support:

.. code-block:: bash

   pip install -e ./tabascal/[gpu]

Running Simulations
-------------------

Simulations are defined by YAML config files and can be launched using:

.. code-block:: bash

   sim-vis -c path/to/config.yaml -st spacetrack_login.yaml

For help:

.. code-block:: bash

   sim-vis -h

Subtracting Satellite-based RFI
-------------------------------

RFI subtraction runs are defined by YAML configuration files and can be run using:

.. code-block:: bash

   tabascal -c path/to/config.yaml -ms path/to/ms/file -st spacetrack_login.yaml

For help:

.. code-block:: bash

   tabascal -h

Space-Track Login
-----------------

In order to get historical orbital elements to predict the positions of satellites in an observation a Space-Track account is needed. 

Your login details should then be saved in a YAML file that is formatted as follows:

.. code-block:: yaml

   username: user@email.com
   password: password1234