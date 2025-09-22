# ISG Pragmatic Similarity Code
## Author: Marcel de Korte, based on work from Andy Segura

This toolkit is primarily intended for computing the pragmatic similarity between a reference sample and a set of target audio samples.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Pragmatic_Similarity_Computation.git
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
- Loading and preprocessing audio files
- Extracting features
- Computing similarities
- Analyzing results

## Dataset

The default dataset for this project is the DRAL dataset. The notebook contains information for how to obtain and organize this data. Alternative datasets can be supported in a similar way.

## Notes

Note that read speech and/or monologues will typically have lower cosine similarity than (in-corpus) dialogue samples, given the differences in speaking patterns and (lack of) pragmatic intent.
