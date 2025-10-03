# Network Traffic Analysis and Classification (TII-SSRC-23)

Repository: https://github.com/AtaQou/DataMiningAndMachineLearning

Goal:
- Understand a network traffic dataset (quick visualizations and stats)
- Make the data smaller so it’s easier to work with
- Train simple models to detect if traffic is normal or malicious, and to predict the traffic type

Important:
- The original dataset (data.csv) is huge and not included in this repo.
- We include a smaller file with the same structure: data_sampled.csv
- Original dataset link (Kaggle): https://www.kaggle.com/datasets/daniaherzalla/tii-ssrc-23

## What’s in this repo

- analysis.py — Full exploratory analysis (prints stats, shows graphs)
- SmallAnalysis.py — Lighter EDA that saves a few key graphs to files
- SampledAnalysis.py — Creates data_sampled.csv from the original data.csv (if you download it)
- KClustering.py — Makes a smaller representative dataset using KMeans (data_clustered.csv)
- AgglomerativeClustering.py — Another clustering method (data_clustered_agglomerative.csv)
- SVMandNeural.py — Trains two models (SVM and a small Neural Network) and prints results
- corolationheatmap.py — Creates a full correlation heatmap across numeric columns
- data_sampled.csv — Small sample included so you can run most things quickly
- requirements.txt — Python libraries needed

Tip: If a script looks for data.csv but you only have data_sampled.csv, you can edit the filename at the top of the script or rename your file accordingly.

## Quick start (with the included sample)

1) Get the code
```

bash git clone https://github.com/AtaQou/DataMiningAndMachineLearning.git cd DataMiningAndMachineLearning``` 

2) Create a virtual environment and activate it
- macOS/Linux:
```

bash python3 -m venv venv source venv/bin/activate``` 
- Windows (PowerShell):
```

bash python -m venv venv venv\Scripts\Activate.ps1``` 

3) Install dependencies
```

bash pip install --upgrade pip pip install -r requirements.txt``` 

4) Explore the sample data (creates and saves some charts)
```

bash python SmallAnalysis.py``` 
or a fuller (more verbose) analysis:
```

bash python analysis.py``` 

5) Generate a correlation heatmap
```

bash python corolationheatmap.py``` 
- This opens a window with the heatmap (if your environment supports it).
- To save it to a file instead of showing it, replace the final line in the script with:
```

python plt.savefig("correlation_heatmap.png", dpi=200, bbox_inches="tight") plt.close()``` 

6) Train and evaluate the models (if clustered data exists)
```

bash python SVMandNeural.py``` 
This prints classification reports that show how well the models predict:
- Label (Normal vs Malicious)
- Traffic Type (category of traffic)

## If you want to regenerate datasets yourself (optional)

Only do this if you want to start from the original big file.

1) Download the original CSV (“data.csv”) from Kaggle and put it in the project folder:
- https://www.kaggle.com/datasets/daniaherzalla/tii-ssrc-23

2) Make a smaller sample (about 1% of the original)
```

bash python SampledAnalysis.py``` 
Produces: data_sampled.csv

3) Create clustered representative datasets (even smaller)
- KMeans:
```

bash python KClustering.py``` 
Produces: data_clustered.csv

- Agglomerative:
```

bash python AgglomerativeClustering.py``` 
Produces: data_clustered_agglomerative.csv

Note: For very large data, these steps can use a lot of memory. If you hit limits, reduce the number of clusters or subsample size inside the scripts.

## What the results mean (plain language)

- We train two simple models:
  - SVM (Support Vector Machine)
  - A small Neural Network
- We test them on two tasks:
  1) Is the traffic normal or malicious? (Label)
  2) What type of traffic is it? (Traffic Type)

Typical outcome observed here:
- Both models do very well at detecting malicious traffic overall.
- Detecting “normal” traffic can be harder because the dataset is very imbalanced.
- For predicting Traffic Type, the Neural Network is usually a bit better across all classes.

You’ll see numbers like “accuracy,” “precision,” “recall,” and “F1-score.” Higher is better. “Macro/weighted F1” help you see performance across all categories, not just the most common ones.

## Common issues and tips

- Can’t open plots? Some environments don’t show windows; use SmallAnalysis.py which saves images to files.
- Memory errors on clustering? Lower the number of clusters in KClustering.py or use the agglomerative script with fewer points.
- Class imbalance: If one class dominates, consider techniques like class weights or different sampling.

## Requirements

- Python 3.10–3.11 recommended
- Install with:
```

bash pip install -r requirements.txt``` 

## Credits

- Dataset: TII-SSRC-23 (Kaggle link above)
- IDE used: PyCharm
```
