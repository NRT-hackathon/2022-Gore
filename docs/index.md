# Documentation for lpdgen

## Recommendations for handling data and running processes
We outline some recommendations for handling the data and training process.
We highly recommend following the recommendations of Pineau et al. outlined
in the paper `Improving Reproducibility in Machine Learning Research`.


### Data organization
Here is a suggested organization of input data and training information.

```
.
├── Data/
│   └── Dataset-1/ 
│       ├── raw_data/ (Data downloaded from the source. Can be one or more images)
│       │   ├── image1.hdf5
│       │   ├── image1-labels.hdf5 (labels needed for supervised tasks)
│       │   ├── image1.hdf5
│       │   └── image2-labels.hdf5
│       ├── training_data/ (Smaller patches generated from raw images for training)
│       │   ├── image1-1.hdf5
│       │   ├── image1-1-labels.hdf5 
│       │   ├── image1-2.hdf5
│       │   ├── image1-2-labels.hdf5
│       │   ├── image2-1.hdf5
│       │   ├── image2-1-labels.hdf5
│       │   ├── image2-2.hdf5
│       │   ├── image2-2-labels.hdf5
│       │   └── training_data.csv (Holds patch coordinates)
│       └── metadata.json (Metadata associated with the entire dataset)
└── Experiments/
    ├── Experiment_1/
    │   ├── notes.txt (Notes related to training run)
    │   ├── experiment_parameters.json (Training hyperparameters)
    │   ├── analysis.ipynb (Analysis of results)
    │   ├── checkpoints/ (Model checkpoints)
    │   │   ├── model_epoch1.pth
    │   │   └── model_epoch2.pth
    │   ├── lightning_logs/ (Training logs for use with tensorboard)
    │   └── model_predictions/ (Output data from inference runs)
    │       ├── predicted-image-1.hdf5
    │       └── predicted-image-2.hdf5
    └── Experiment_2/
        ├── notes.txt
        ├── experiment_parameters.json
        ├── analysis.ipynb
        ├── checkpoints/
        │   ├── model_epoch1.pth
        │   └── model_epoch2.pth
        ├── lightning_logs/
        └── model_predictions/
            ├── predicted-image-1.hdf5
            └── predicted-image-2.hdf5

```
