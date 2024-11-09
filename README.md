# ackl

### Publication

    Analytical chemistry kernel library for spectroscopic profiling data, Food Chemistry Advances, Volume 3, 2023, 100342, ISSN 2772-753X, https://doi.org/10.1016/j.focha.2023.100342.

### PyPI (Python Package Index) repository

https://pypi.org/project/ackl/ 

### Reproducible Code-Ocean capsule

https://doi.org/10.24433/CO.4614220.v2


# Install (Ubuntu Env Setup)

```
!apt-get install r-base r-base-dev ffmpeg libsm6 libxext6 

!pip install rpy2
!pip install qsi==0.3.9
!pip install ackl==1.0.2
!pip install cla==1.1.4
!pip install opencv-python

# Post-install script

#!/usr/bin/env bash
set -e

Rscript -e 'install.packages("ECoL")'
```

# Compile

```
python -m build --wheel  
python -m pyc_wheel x.whl  [optional]
```

# Use

### Kernel Response Patterns

```
import ackl.metrics
ackl.metrics.linear_response_pattern(20)
```

### Run Kernels on Target Dataset

```
_, dics, _ = ackl.metrics.classify_with_kernels(X, y,embed_title = False)
```

Show the result as HTML table and bar charts: 

```
html_str = ackl.metrics.visualize_metric_dicts(dics, plot = True)
display(HTML( html_str ))
```
