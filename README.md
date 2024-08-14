# SYCL Tutorial 2024

**Note:** This tutorial is currently under development and is being updated continually. Please check back regularly for the latest updates.

## Creating your account at the Intel Developer Cloud

These code examples can be executed at the Intel Dev Cloud. If you don't have an account, point your browser to
https://www.intel.com/content/www/us/en/developer/tools/devcloud/services.html
and click on the "Sign Up" button.

You will be asked to enter an email. The system will send a verification code to your email account. Enter the verification code and now the system will ask you to enter and verify a password. After this step you must Terms and Conditions and then you will be in.

From the three possible options you are given (Learn, Evaluate, Deploy), select Learn and press its corresponding Get Started button.

In the new screen, use the leftmost button "Connect to a GPU". After a few second, a window will pop up letting you know that "Your JupyterLab access is ready". Press the Launch button. It may take a few seconds or up to a minute to connect. At this point you will be presented with a JupyterLab interface that may be already familiar to you.

## Cloning this repo

You will need to clone this repo in the computing environment for this tutorial. Use the following command line:

```
git clone https://github.com/SYCLTutorials/Intro2024.git
```

If you are running on Intel Dev Cloud, launch a terminal and execute the above `git clone` command. This will clone this repository in your home directory. Navigate to directory "Intro2024" and the contents of this repository will be there.



## What's inside?


## Jupyter notebooks

If you are contributing any code in IPython/Jupyter notebooks, *please*
install the `nbstripout` extension (available e.g. on
`github <https://github.com/kynan/nbstripout#installation>`_ and
`PyPI <https://pypi.org/project/nbstripout/>`_).  After installing,
activate it for this project by running:

```
   nbstripout --install --attributes .gitattributes
````

from the top-level repository directory.  Please note that that
``nbstripout`` will not strip output from cells with the metadata fields
``keep_output`` or ``init_cell`` set to ``True``, so use these fields
judiciously.  You can ignore these settings with the following command:

```

   git config filter.nbstripout.extrakeys '\
      cell.metadata.keep_output cell.metadata.init_cell'
```
(The keys ``metadata.kernel_spec.name`` and
``metadata.kernel_spec.display_name`` may also be useful to reduce diff
noise.)

Nonetheless, it is highly discouraged to contribute code in the form of
notebooks; even with filters like ``nbstripout`` they're a hassle to use
in version control. Use them only for tutorials or *stable* examples that
are either meant to be run *interactively*.

## Acknowledgements
This research used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357. Argonne National Laboratory’s work was supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357.


