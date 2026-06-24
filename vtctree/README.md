# VTC-Tree (Verifying Temporal Cuts Decision Tree)

VTC-Tree is an extension of the TC-Tree classifier for temporal data. The method introduces verifying temporal cuts, which use alternative attribute-based splits to provide additional evidence supporting classification decisions.

## Contents

- VTC-Tree classifier implementation (`vtctree.py`)
- Example usage script (`demo_vtctree.py`)
- Sample datasets (`data/`)

## Usage

See `demo_vtctree.py` for a complete workflow including:
- loading temporal data,
- training a VTC-Tree model,
- making predictions,
- evaluating classification performance.

## Description

The proposed approach extends the TC-Tree framework by incorporating verifying temporal cuts during tree construction and classification. The additional cuts are intended to increase the reliability of classification decisions by exploiting information from alternative temporal cuts.

## Citation

A publication describing VTC-Tree is currently under preparation. Until it becomes available, please cite this repository.

## License

This software is distributed under the BSD 3-Clause License. See the repository `LICENSE` file for details.