:: Define experiment name
:: The below data preparation functions don't output anything but still require an output to be specified (a harmless bug of the data module)
:: So to avoid generating an empty output folder, use 'dataset_analysis' since the data analysis functions do output files
set experiment=dataset_analysis

:: Run data module
python -m gmtames data --generateBaseDatasets --generateApplicabilityDomainDatasets --output %experiment%