# ackl

Analytical Chemisty Kernels Library

# Install (CodeOcean Ubuntu Env Setup)

```
!apt-get install r-base r-base-dev ffmpeg libsm6 libxext6 

!pip install rpy2
!pip install qsi==0.3.9
!pip install ackl==1.0.2
!pip install cla==1.1.3
!pip install opencv-python

# Post-install script

#!/usr/bin/env bash
set -e

Rscript -e 'install.packages("ECoL")'
```

# Use

## Kernel Response Patterns

```
import ackl.metrics
ackl.metrics.linear_response_pattern(20)
```

## Run Kernels on Target Dataset

```
dics = ackl.metrics.preview_kernels(X, y,embed_title = False, scale = False, logplot = False)
```