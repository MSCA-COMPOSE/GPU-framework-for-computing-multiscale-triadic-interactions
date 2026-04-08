Figures

This directory contains example figures generated from the reduced public test case distributed with this repository, together with the corresponding plotting script.

The current examples are intended as a minimal demonstration of the post-processing workflow and show how to import the reduced POD and triadic outputs and visualize:

the temporal coefficients obtained from the POD step;
the triadic-interaction tensor represented as a reduced 3D cube.

These figures are not meant to reproduce the full production dataset. They are included as lightweight examples to help users verify that the public workflow and plotting scripts are working correctly.

Running the plotting script
Enter the project directory:

cd GPU-framework-for-computing-multiscale-triadic-interactions

Create a dedicated conda environment for the plotting step:

conda create -n triadic_plot python=3.11 numpy=1.26 pandas=2.1.4 pyarrow=14.0.2 matplotlib

conda activate triadic_plot

If you want to run the plotting script from Spyder, install the matching Spyder kernel in the same environment:

conda install spyder-kernels=3.1

If needed, install Spyder in that environment and launch it from there:

conda install spyder -y

spyder

Run the plotting script located in this directory.
Note on the plotting environment

The reduced POD outputs are distributed in Parquet directory format. In our tests, reading these files was reliable with Python 3.11, pandas 2.1.4, and pyarrow 14.0.2.

Using a dedicated environment with these versions is therefore recommended for the plotting and post-processing script included in this directory.
