# ALIGNN — Formation Energy Prediction

A Python-based tool for predicting the formation energy of Metal–Organic Framework (MOF) CIF structures using [ALIGNN (Atomistic Line Graph Neural Network)](https://github.com/usnistgov/alignn).

The script uses the pretrained `jv_formation_energy_peratom_alignn` model to predict formation energy in eV/atom. It processes multiple CIF files in parallel, generates CSV reports, and copies structures with negative predicted formation energy into the output directory.

## Features

- Predicts formation energy using pretrained ALIGNN.
- Processes multiple CIF files in parallel using `ProcessPoolExecutor`.
- Converts CIF structures into the format required for ALIGNN prediction.
- Filters structures with predicted formation energy below 0 eV/atom.
- Generates CSV reports containing formation-energy predictions.
- Copies qualifying CIF files into the output directory.

## Requirements

- Python 3.9 (recommended starting point; verify compatibility with the installed dependencies).
- Conda or Miniconda.
- ALIGNN source code.
- The pretrained formation-energy model ZIP file.
- MOF CIF input files.
- Sufficient RAM for parallel model inference.

## 1. Clone the Repository

```bash
git clone https://github.com/grp27-kmof/Alignn.git
cd Alignn
```

## 2. Create a Conda Environment

Create a dedicated Conda environment:

```bash
conda create -n alignn python=3.9 -y
```

## 3. Activate the Environment

```bash
conda activate alignn
```

Make sure the environment is activated before installing dependencies or running the script.

## 4. Clone and Install ALIGNN

Clone the official ALIGNN repository inside your project directory:

```bash
git clone https://github.com/usnistgov/alignn.git
```

Install ALIGNN and its dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ./alignn
```

This installs ALIGNN in editable mode and installs its declared dependencies. Depending on your operating system and Python environment, you may need compatible PyTorch and other scientific-library versions.

## 5. Add the Pretrained Model

The script uses the following pretrained model:

`jv_formation_energy_peratom_alignn.zip`

Place the model ZIP file inside:

```text
alignn/alignn/jv_formation_energy_peratom_alignn.zip
```

The model is loaded directly from this ZIP file by `main.py`.

The bandgap model `jv_mbj_bandgap_alignn.zip` is not required for the current script because it does not perform bandgap prediction.

Verify that the formation-energy model exists:

```bash
ls alignn/alignn/jv_formation_energy_peratom_alignn.zip
```

**Note:** Ensure that the model file is available locally before running the script. Do not assume that cloning the ALIGNN source repository automatically provides this particular pretrained model ZIP.

## 6. Add Input CIF Files

Place the MOF CIF files you want to analyze in your chosen input directory.

Open `main.py` and locate:

```python
CIF_DIR = r"/home/ubuntu/Documents/MOF_ANODE27/Alignn1/input"
```

Change the path to the directory containing your CIF files.

For example:

```python
CIF_DIR = r"/path/to/your/input"
```

The script processes `.cif` files directly inside this directory, not files in nested subdirectories.

## 7. Configure Parallel Execution

In `main.py`, locate the following settings:

```python
max_workers = 25
model_name = "jv_formation_energy_peratom_alignn"
```

The script loads a separate ALIGNN model in each worker process. This allows multiple CIF files to be processed concurrently.

This configuration was designed for a **32-core workstation using 25 workers**. Adjust `max_workers` according to your system's CPU resources and available RAM.

Each worker loads its own model, so memory usage can be substantial. If your system runs out of memory, reduce the worker count.

## 8. Run the Script

From the project root, execute:

```bash
python main.py
```

The script will:

1. Find the CIF files in the configured input directory.
2. Start a pool of worker processes and load the pretrained model in each worker.
3. Read each CIF structure using ASE.
4. Convert each structure to POSCAR format temporarily and load it using JARVIS.
5. Construct the graphs required by ALIGNN.
6. Predict formation energy in eV/atom.
7. Save all prediction results to a CSV file.
8. Copy structures with predicted formation energy below 0 eV/atom into the output directory.
9. Save the filtered predictions to a separate CSV file.

Example terminal output (illustrative):

```text
Found 100 CIF files.

======================================================================
Loading ALIGNN Formation Energy model across 25 parallel workers...
This may take a moment...
======================================================================

======================================================================
[1/100] Processing MOF_001.cif
Formation Energy : -0.1234 eV/atom
Passed filter.

======================================================================
[2/100] Processing MOF_002.cif
Formation Energy : 0.0456 eV/atom

======================================================================
Total MOFs       : 100
Filtered MOFs    : 42

All Results CSV  : alignn_results.csv
Filtered CSV     : filtered_mofs.csv
Filtered CIFs    : /path/to/repository/output
======================================================================
```

The example filenames, predicted values, and counts are illustrative and are not measured results. The actual output depends on the input structures and model predictions. Processing errors are printed for individual files.

## 9. Formation-Energy Filtering

The script uses the following criterion:

formation energy < 0

Structures satisfying this condition are copied into the output directory.

This is the screening criterion implemented in the script. A negative predicted formation energy alone does not establish thermodynamic stability against all competing phases; further validation may be necessary.

## 10. Output Files

### `alignn_results.csv`

Contains the predictions for processed CIF files, with the following columns:

| Column | Description |
|---|---|
| `cif_file` | Input CIF filename |
| `Formation Energy` | Predicted formation energy in eV/atom |

### `filtered_mofs.csv`

Contains only structures whose predicted formation energy is below 0 eV/atom, with the same two columns.

### `output/`

Contains copies of the original CIF files that pass the formation-energy filter.

**Note:** CIF files that fail conversion or prediction are recorded with an empty formation-energy value in `alignn_results.csv`. They are not included in `filtered_mofs.csv`. Check the terminal output for their error messages.


## Acknowledgements

This project uses:

- [ALIGNN](https://github.com/usnistgov/alignn) for formation-energy prediction.
- [ALIGNN research paper](https://www.nature.com/articles/s41524-021-00650-1).
- [ALIGNN follow-up paper](https://arxiv.org/abs/2201.08348).
- [ASE](https://wiki.fysik.dtu.dk/ase/) for structure reading and writing.
- [JARVIS](https://jarvis.nist.gov/) for atomistic structure representation.
- [PyTorch](https://pytorch.org/) for model inference.
