# Code for performing automatic analyis on the **KUG Dastgāh Corpus**

This repository contains the required code and annotations file for performing preliminary automatic analysis on a dataset created from the **KUG Dastgāh Corpus** (**KDC**).

The analyses are based on [Essentia](https://essentia.upf.edu/)<sup>1</sup>, an open source library of algorithms for audio analysis. They have been written in a [Jupyter Notebook](https://jupyter.org/) for better reproducibility.

Using the provided metadata, the code provided in this repository performs the following analyses and plots:

- Pitch-track extraction</br>The pitch track can be plotted over a spectrogram with a horizontal line marking the pitch of the šāhed. It also generates a text file with the pitch track that can be imported in [Sonic Visualiser](https://www.sonicvisualiser.org/) as annotation.

- Pitch histogram computation</br>The pitch histogram can be plotted folded into an octave with the šāhed as the first peak, or unfolded, with the peak for the šāhed highlighted.

- Vibrato analysis</br>A plot can be produced containing the pitch track, the vibrato frequency and the vibrato extent for those regions where vibrato was automatically detected.

- Loudness analysis</br>A plot can be produced containing the pitch track and the loudness track.

- Pitch of the first and last note</br>It generates text files with the detected notes which can be imported in [Sonic Visualiser](https://www.sonicvisualiser.org/) as annotations.


For all the analyses, a text file is generated with the specific numeric results of each analysis (except for loudness)


 ## Content

 This repository contains the following files:

 * The Jupyter Notebook `KDC-Analysis.ipynb` is the main file with the code for running the proposed analyses.

 * The `essentiaUtils.py` file defines some Python functions using algorithms from Essentia that are used in the previous notebook.

 * The `intonation.py` file is borrowed from the [intonation GitHub repository](https://github.com/gopalkoduri/intonation) by Gopala Koduri<sup>2</sup>, and is used in the notebook for computing pitch histograms. Small modifications have been added to adapt the code to Python 3 and customize the resulting plots.

 * The `KDC-data.csv` contains the required metadata and annotations for running the notebook.
  - The column `Shahed` contain pitch values for the šāhed of each file, manually verified by the authors
  - The columns `Min_freq`, `Max_freq` and `f0_cf` contain manually verified values for the minimum frequency, maximum frequency and pitch confidence threshold for extracting pitch track

 ## Access to the **KDC**
 Since the **KDC** contains a collection of commercial recordings, it cannot be made publicly available. If you are interesting of using the corpus for research purposes, con can contact Babak Nikzat (b.nikzat at kug.ac.at).

 ## How to use

 In order to run the code, you need to downloaded either using [Git](https://git-scm.com/) or as a zip file. Then you also need to install the following software:

 * Essentia: in its website you can find a [guide for installing Essentia](https://essentia.upf.edu/installing.html). Except for Linux users, this can be a bit complicated. For developing this repository, a computer with Windows 10 has been used. According to our experience, the easiest way to have Essentia running in a Windows system is installing a Linux distribution, following this [official installation guide from Microsoft](https://docs.microsoft.com/de-de/windows/wsl/install-win10#install-your-linux-distribution-of-choice) (for this repository we used Ubuntu 18.04 LTS). Once the Linux distribution is installed, Essentia can be installed there by running

 >`pip3 install essentia`

 from the terminal (installing `pip` might be required, follow the instructions that will appear). The rest of the following software should be also installed in the Linux distribution, and the code also should be run from there.

 * Jupyter: in its website you can find a [guide for installing Jupyter](https://jupyter.org/install)

 * The dependencies contained in the requirements file. In order to do so, using a terminal go to the directory where you downloaded this repository and which contains the `requirements.txt` file and run the following command:

 > `pip3 install -r requirements.txt`

 To run the Jupyter Notebook, just open a terminal, go to the directory where the notebook is stored, and run the command

 > `jupyter-notebook`

 The notebook will open in your default internet browser.

 # License
 The content of this repository is licensed under the terms of



---
\[1\] Bogdanov D, Wack N, Gómez E, Sankalp G, Herrera P, Mayor O, Roma G, Salamon J, Zapata J, Serra X. Essentia: an audio analysis library for music information retrieval. In: Britto A, Gouyon F, Dixon S, editors. *14th Conference of the International Society for Music Information Retrieval (ISMIR)*; 2013 Nov 4-8; Curitiba, Brazil. [place unknown]: ISMIR; 2013. p. 493-8. [http://hdl.handle.net/10230/32252](http://hdl.handle.net/10230/32252)


\[2\] Koduri GK, Ishwar V, Serrà J, Serra X. Intonation analysis of rāgas in Carnatic music. *Journal of New Music Research*. 2014;43(01):73–94. [http://hdl.handle.net/10230/25676](http://hdl.handle.net/10230/25676)
