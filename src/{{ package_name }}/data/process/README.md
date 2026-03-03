# Data Preprocessing

The subpackage contains initial preprocessing of the raw data.
It is exclusively concerned with removing rows and outliers.

Any operations that impute or transform without modifying the data shape is part **not** part of the preprocessing but is performed inside the ML-pipeline.

Ideally, there is little to no extra processing, but missing values and outliers are imputed in the ML-pipeline instead.
