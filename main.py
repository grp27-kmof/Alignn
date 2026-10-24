#!/usr/bin/env python3

import glob
import csv
import os
import sys
import json
import zipfile
import tempfile
import shutil
import uuid
import concurrent.futures

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(BASE_DIR, "alignn"))

import torch
from ase.io import read, write
from jarvis.core.atoms import Atoms
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.graphs import Graph

ALIGNN_DIR = os.path.join(
    BASE_DIR,
    "alignn",
    "alignn",
)
MODEL_ZIPS = {
    "jv_formation_energy_peratom_alignn": os.path.join(
        ALIGNN_DIR,
        "jv_formation_energy_peratom_alignn.zip",
    ),
}

CIF_DIR = r"/home/ubuntu/Documents/MOF_ANODE27/Alignn1/input"

OUTPUT_CSV = "alignn_results.csv"
FILTERED_CSV = "filtered_mofs.csv"

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DEVICE = torch.device("cpu")

global_fe_model = None

def load_model(model_name):

    zp_path = MODEL_ZIPS[model_name]

    if not os.path.isfile(zp_path):
        raise FileNotFoundError(f"Model zip not found:\n{zp_path}")

    with zipfile.ZipFile(zp_path, "r") as zp:

        names = zp.namelist()

        model_file = next(
            (n for n in names if "best_model.pt" in n),
            None,
        )

        if model_file is None:
            model_file = next(
                n
                for n in names
                if "checkpoint_" in n and n.endswith(".pt")
            )

        config_file = next(
            n
            for n in names
            if n.endswith("config.json")
        )

        config = json.loads(zp.read(config_file))
        model_bytes = zp.read(model_file)

    fd, temp_model = tempfile.mkstemp(suffix=".pt")

    try:

        with os.fdopen(fd, "wb") as f:
            f.write(model_bytes)

        model = ALIGNN(
            ALIGNNConfig(**config["model"])
        )

        checkpoint = torch.load(
            temp_model,
            map_location=DEVICE,
        )

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

    finally:

        if os.path.exists(temp_model):
            os.remove(temp_model)

    model.to(DEVICE)
    model.eval()

    return model


def predict(model, atoms, cutoff=8.0, max_neighbors=12):

    graph_data = Graph.atom_dgl_multigraph(
        atoms,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
    )

    g = graph_data[0]
    lg = graph_data[1]

    lattice = (
        torch.tensor(
            atoms.lattice_mat,
            dtype=torch.float32,
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        pred = model(
            [
                g.to(DEVICE),
                lg.to(DEVICE),
                lattice,
            ]
        )

    return float(pred.cpu().numpy().flatten()[0])


def init_worker(model_name):
    global global_fe_model

    torch.set_num_threads(1)

    global_fe_model = load_model(model_name)


def process_cif(cif_file):
    name = os.path.basename(cif_file)

    poscar_name = f"POSCAR_{uuid.uuid4().hex}"
    
    try:
        ase_atoms = read(cif_file)
        write(
            poscar_name,
            ase_atoms,
            format="vasp",
        )
        atoms = Atoms.from_poscar(poscar_name)
    except Exception as e:
        return {
            "cif_file": name,
            "Formation Energy": None,
            "error": f"Conversion Error: {e}"
        }
    finally:
        if os.path.exists(poscar_name):
            os.remove(poscar_name)

    try:
        fe = predict(
            global_fe_model,
            atoms,
        )
        return {
            "cif_file": name,
            "Formation Energy": fe,
            "error": None
        }
    except Exception as e:
        return {
            "cif_file": name,
            "Formation Energy": None,
            "error": f"Formation Energy Error: {e}"
        }


def main():

    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cif_files = sorted(
        glob.glob(os.path.join(CIF_DIR, "*.cif"))
    )

    total = len(cif_files)

    if total == 0:
        print("No CIF files found.")
        return

    print(f"Found {total} CIF files.\n")

    max_workers = 25
    model_name = "jv_formation_energy_peratom_alignn"

    print("=" * 70)
    print(f"Loading ALIGNN Formation Energy model across {max_workers} parallel workers...")
    print("This may take a moment...")
    print("=" * 70)

    results = []
    filtered_results = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers, 
        initializer=init_worker, 
        initargs=(model_name,)
    ) as executor:

        future_to_cif = {
            executor.submit(process_cif, cif_file): cif_file 
            for cif_file in cif_files
        }

        for i, future in enumerate(concurrent.futures.as_completed(future_to_cif), start=1):
            cif_file = future_to_cif[future]
            name = os.path.basename(cif_file)
            
            print("=" * 70)
            print(f"[{i}/{total}] Processing {name}")
            
            try:
                res = future.result()
                
                fe = res["Formation Energy"]
                error = res["error"]
                
                if error:
                    print(error)
                else:
                    print(f"Formation Energy : {fe:.4f} eV/atom")
                
                record = {
                    "cif_file": res["cif_file"],
                    "Formation Energy": fe,
                }
                results.append(record)

                if fe is not None and fe < 0:
                    filtered_results.append(record)
                    shutil.copy2(
                        cif_file,
                        os.path.join(OUTPUT_DIR, name),
                    )
                    print("Passed filter.")
                    
            except Exception as exc:
                print(f"File {name} generated an unexpected exception: {exc}")
                
            print()

    results.sort(key=lambda x: x["cif_file"])
    filtered_results.sort(key=lambda x: x["cif_file"])

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cif_file",
                "Formation Energy",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    with open(
        FILTERED_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cif_file",
                "Formation Energy",
            ],
        )

        writer.writeheader()
        writer.writerows(filtered_results)

    print("=" * 70)
    print(f"Total MOFs       : {len(results)}")
    print(f"Filtered MOFs    : {len(filtered_results)}")
    print()
    print(f"All Results CSV  : {OUTPUT_CSV}")
    print(f"Filtered CSV     : {FILTERED_CSV}")
    print(f"Filtered CIFs    : {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()