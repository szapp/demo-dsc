# Data Loading

This subpackage orchestrates fetching the raw data from multiple data sources and joining it.

## SQL Queries

The process follows the concept of a feature store.
Features and Target are defined in individual, parameterized SQL queries.
A file `index.sql` is required to define the identifier column(s) on which the features and target data are left joined.

## Caching

The joined and validated data is cached to disk.
Subsequent calls to the central loading function return the cached result until the cache expires or purged by a forced recompute.
