# Fake Profile Detection

This project aims to detect fake profiles using a Random Forest classifier. The project is organized into scripts for data generation, model training, and model evaluation.

## Folder Structure

fake_profile_detection/
 |tests
│
├── data/
│ └── profile_data.csv
│
├── scripts/
│ ├── generate_data.py
│ ├── train_model.py
│ └── evaluate_model.py
│
└── requirements.txt 

## Setup

### Prerequisites

- Python 3.x
- `pip` (Python package installer)

### Installation

1.Install the required Python packages:
pip install -r requirements.txt

2.cd scripts
python generate_data.py
The generate_data.py script generates a synthetic dataset and saves it to the data folder


python train_model.py
The train_model.py script trains a Random Forest model using the synthetic dataset and saves the trained model to the data folder

python evaluate_model.py
The evaluate_model.py script evaluates the trained model using the test set and prints the accuracy and classification report

3.although tests csv are not used they are to demostrate data division

4.My test results are stored in results.csv.

5.all done !!!!
6.You can configure test.py and test-data/values.csv to test with ur own dataset
