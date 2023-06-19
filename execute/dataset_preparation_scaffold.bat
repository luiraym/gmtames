:: Define test split strategy
set testsplit=scaffold

:: Set experiment name
:: The below data preparation functions don't output anything but still require an output to be specified (a harmless bug of the data module)
set experiment=mtg_experiment_%testsplit%

:: Run data module
python -m gmtames data --generateBaseDatasets --generateApplicabilityDomainDatasets --testsplit %testsplit% --output %experiment%