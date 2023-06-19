# Using batch files to execute the mechanistic task grouping experiments on Windows

## Execution steps
1. Open Python terminal and start in the main gmtames directory (one level above here)

2. Activate the gmtames conda environment
```bash
conda activate gmtames
```

3. Prepare datasets
```bash
execute\dataset_preparation_maxmin.bat
```

4. Run ad_calc in MATLAB

5. Run experiments
```bash
execute\mtg_experiment_maxmin.bat
```

6. Repeat steps 3 - 5 for both MaxMin and scaffold datasets

## Last confirmed technical requirements
| | |
| --- | --- |
| **Operating system** | Windows 10 Education, version 22H2, build number 19045.3086 |
| **Hardware accelerator** | NVIDIA TITAN V, driver version 516.94, CUDA version 11.7 |