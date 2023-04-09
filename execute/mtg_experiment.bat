setlocal enabledelayedexpansion
:: See https://www.robvanderwoude.com/variableexpansion.php for variable expansion tutorial

:: Define experiment name
set experiment=mtg_experiment

:: Define task groupings to evaluate
:: ST groupings
set groupings="TA100"
set groupings=%groupings%;"TA100_S9"
set groupings=%groupings%;"TA102"
set groupings=%groupings%;"TA102_S9"
set groupings=%groupings%;"TA104"
set groupings=%groupings%;"TA104_S9"
set groupings=%groupings%;"TA1535"
set groupings=%groupings%;"TA1535_S9"
set groupings=%groupings%;"TA1537"
set groupings=%groupings%;"TA1537_S9"
set groupings=%groupings%;"TA1538"
set groupings=%groupings%;"TA1538_S9"
set groupings=%groupings%;"TA97"
set groupings=%groupings%;"TA97_S9"
set groupings=%groupings%;"TA98"
set groupings=%groupings%;"TA98_S9"

:: uMT grouping
set groupings=%groupings%;"TA100,TA100_S9,TA102,TA102_S9,TA104,TA104_S9,TA1535,TA1535_S9,TA1537,TA1537_S9,TA1538,TA1538_S9,TA97,TA97_S9,TA98,TA98_S9"

:: gMT-mtg groupings
:: Substitution strains
set groupings=%groupings%;"TA100,TA100_S9,TA102,TA102_S9,TA104,TA104_S9,TA1535,TA1535_S9"
:: Frameshift strains
set groupings=%groupings%;"TA1537,TA1537_S9,TA1538,TA1538_S9,TA97,TA97_S9,TA98,TA98_S9"
:: Non-S9 strains
set groupings=%groupings%;"TA100,TA102,TA104,TA1535,TA1537,TA1538,TA97,TA98"
:: S9 strains
set groupings=%groupings%;"TA100_S9,TA102_S9,TA104_S9,TA1535_S9,TA1537_S9,TA1538_S9,TA97_S9,TA98_S9"
:: Non-S9 substitution strains
set groupings=%groupings%;"TA100,TA102,TA104,TA1535"
:: Non-S9 frameshift strains
set groupings=%groupings%;"TA1537,TA1538,TA97,TA98"
:: S9 substitution strains
set groupings=%groupings%;"TA100_S9,TA102_S9,TA104_S9,TA1535_S9"
:: S9 frameshift strains
set groupings=%groupings%;"TA1537_S9,TA1538_S9,TA97_S9,TA98_S9"

:: Run MTG experiment
for %%g in (%groupings%) do (
	set grouping=%%g
	set grouping=!grouping:~1,-1%!
	python -m gmtames mtg --tasks !grouping! --output %experiment% --device titanv
)

:: Calculate experiment results
python -m gmtames results %experiment%