# SYCL Tutorial 2024

**Note:** This tutorial is currently under development and is being updated continually. Please check back regularly for the latest updates.

## What's inside?


Jupyter notebooks
^^^^^^^^^^^^^^^^^

If you are contributing any code in IPython/Jupyter notebooks, *please*
install the `nbstripout` extension (available e.g. on
`github <https://github.com/kynan/nbstripout#installation>`_ and
`PyPI <https://pypi.org/project/nbstripout/>`_).  After installing,
activate it for this project by running:

.. code:: shell

   nbstripout --install --attributes .gitattributes

from the top-level repository directory.  Please note that that
``nbstripout`` will not strip output from cells with the metadata fields
``keep_output`` or ``init_cell`` set to ``True``, so use these fields
judiciously.  You can ignore these settings with the following command:

.. code:: shell

   git config filter.nbstripout.extrakeys '\
      cell.metadata.keep_output cell.metadata.init_cell'

(The keys ``metadata.kernel_spec.name`` and
``metadata.kernel_spec.display_name`` may also be useful to reduce diff
noise.)

Nonetheless, it is highly discouraged to contribute code in the form of
notebooks; even with filters like ``nbstripout`` they're a hassle to use
in version control. Use them only for tutorials or *stable* examples that
are either meant to be run *interactively*.

## Acknowledgements
This research used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357. Argonne National Laboratory’s work was supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357.


