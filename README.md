# ISG Pragmatic Similarity Code
## Author: Marcel de Korte, based on code by Andy Segura

This code can be used to compute the pragmatic similarity between a recorded utterance and a set of reference audio samples, and have it return the most similar utterance.  It is thus a re-release of the Pragmatically Similar Utterance Finder Demonstration (Ward and Segura, Interspeech 2024). Its main difference is the easier install process: the key dataset is already included and does not need a separate install, and control is via a Jupyter notebook, rather than via a user interface. This allows for easy experimentation to understand the workings of the similarity metric. However, this version lacks the ability to "eavesdrop" on a human-human conversation and pull out live utterances to use as search seeds. It is therefore less suitable for live demos (since speech produced in response to a prompt is generally not very similar to anything in the reference set, which is all taken from live dialogs, and exhibits spontaneous speaking patterns and a rich diversity of pragmatic intents).

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

Note that read speech and/or monologues will typically lead to lower cosine similarities when compared to a dialog corpus like DRAL than (in-corpus) dialog samples, given the differences in speaking patterns and (lack of) pragmatic intent.
