# Data

This subpackage contains the data pipeline.
The process is similar to ETL:

1. **Load** and join raw data from data sources.
1. **Validate** raw data model to confirm data integrity.
1. **Pre-process** data to remove rows and outliers.
1. **Validate** processed data model to convert to ML-conform data types.

The processing is only concerned with removing rows.
Any processing beyond that is part of the transformations in the ML-pipeline.
