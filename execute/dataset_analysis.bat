:: Define experiment name
set experiment=dataset_analysis

:: Run data module
python -m gmtames data --describeBaseDatasets --correlateBaseDatasets --output %experiment%