# ISG Pragmatic Similarity Code
## Author: Marcel de Korte, revised for generality and easy of use from Andy Segura

This toolkit is intended for computing the pragmatic similarity between a reference sample and a set of target audio samples.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mdekorte/Pragmatic_Similarity_Computation.git
cd Pragmatic_Similarity_Computation
```

2. Create and activate a virtual environment (optional but recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

Alternatively, use a conda or mamba environment.

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

The easiest way to get familiar with the code is through our Jupyter notebook tutorial:

1. Launch Jupyter:
```bash
jupyter notebook
```

2. Navigate to `notebooks/Pragmatic_Similarity_Tutorial.ipynb`

The tutorial walks you through:
- How to load the audio files (optional)
- Extracting features for target sentence
- Computing similarities between target sentence and the corpus for comparison
- Basic perceptual analysis

## Dataset

The default dataset used is the DRAL dataset. The notebook contains information for how to obtain and organize this data. Alternative datasets can be supported in a similar way.

## Notes

Note that read speech and/or monologues will typically have lower cosine similarity than (in-corpus) dialogue samples, given the differences in speaking patterns and (lack of) pragmatic intent.
