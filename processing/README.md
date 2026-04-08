# Figures

This directory contains example figures generated from the reduced public test case distributed with this repository, together with the corresponding plotting script.

The current examples are intended as a minimal demonstration of the post-processing workflow and show how to import the reduced POD and triadic outputs and visualize:

* the modal energy distribution and cumulative energy content,
* the temporal coefficients obtained from the POD step,
* the triadic-interaction tensor represented as a reduced 3D cube.

These figures are not meant to reproduce the full production dataset. They are included as lightweight examples to help users verify that the public workflow and plotting scripts are working correctly.

## Running the plotting script

1. Enter the project directory:

   ```bash
   cd GPU-framework-for-computing-multiscale-triadic-interactions

2. Create a dedicated conda environment for the plotting step:

   ```bash
   conda create -n triadic_plot python=3.11 numpy=1.26 pandas=2.1.4 pyarrow=14.0.2 matplotlib
   conda activate triadic_plot

3. If you want to run the plotting script from Spyder, install the matching Spyder kernel in the same environment:

   ```bash
   conda install spyder-kernels=3.1

4. If needed, install Spyder in that environment and launch it from there:

   ```bash
   conda install spyder -y
   spyder

5. Run the plotting script located in this directory.

## Running the plotting script
The reduced POD outputs are distributed in Parquet directory format. In our tests, reading these files was reliable with:

* Python 3.11
* pandas 2.1.4
* pyarrow 14.0.2

Using a dedicated environment with these versions is therefore recommended for the plotting and post-processing script included in this directory.

   
