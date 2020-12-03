### Requirements

- conda
- Python 3.7 or higher
- open-cv version 4.4.0.46

### Installation

```bash
# If you need to install conda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh


# Create env with requirements
conda env create -f environment.yml

# activate the conda environment
source activate tempAss2

```

### Running code
```bash
# Feature Detection and Matching without outlier detection
python main.py False

# Feature Detection and Matching with outlier detection using threshold of ratioThreshold of 0.95 and distThreshold of 30
python main.py True

# Feature Detection and Matching with outlier detection using threshold of ratioThreshold of 0.8 and distThreshold of 50
python main.py True --ratioThreshold 0.8 --distThreshold 50

# Feature Detection and Matching with outlier detection and print grid search results
python main.py True --gridSearch True

# Feature Detection and Matching with outlier detection and prints per image statistic on training set
python main.py True --perImageStatistic True

# Feature Detection and Matching without outlier detection and saves test detector features in an xlsx
python main.py False --saveDetectorFeaturesOnTestSet True

# Feature Detection and Matching without outlier detection and training depth results in an xlsx
python main.py False --saveTrainDepthResults True
```