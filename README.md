# Tutorial of machine learning for emulating a Lorenz model

### Content of this repository
- [crash_course.pdf](crash_course.pdf): slide of the lecture given at [the crash course on DA](https://events.nersc.no/event/3rd-summer-school-crash-course-data-assimilation-theoretical-foundations-and-advanced) in June 2021 
- [L63_demonstrator.ipynb](L63_demonstrator.ipynb): notebook implemented a small tutorial on model emulation using machine learning

### Instructions for working on the cloud (recommended)

Run the tutorial in a cloud computing provider:

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/nansencenter/ml-crashcourse/blob/main/L63_demonstrator.ipynb)
  (requires Google login)

### Instructions for working locally

You can also run this notebook on your own (Linux/Windows/Mac) computer.
This is a bit snappier than running them online.

1. **Prerequisite**: Python>=3.7.  
   If you're not a python expert:  
   1a. Install Python via [Anaconda](https://www.anaconda.com/download).  
   1b. Use the [Anaconda terminal](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda)
       to run the commands below.  
   1c. (Optional) [Create & activate a new Python environment](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-environments).
       If the installation (below) fails, try doing step 1c first.

2. **Install**:  
   Run these commands in the terminal (excluding the `$` sign):  
   `$ git clone https://github.com/nansencenter/ml-crashcourse.git`  
   `$ pip install -r ml-crahscourse/requirements.txt`  

3. **Launch the Jupyter notebooks**:  
   `$ jupyter-notebook`  
   This will open up a page in your web browser that is a file navigator.  
   Enter the folder `ml-craashcourse`, and click on the tutorial `L63_demonstrator.ipynb`.

<!-- markdownlint-disable-file heading-increment -->
