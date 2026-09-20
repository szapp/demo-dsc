-- This SQL determines ont of the identifiers that other SQLs are based on
with
base (n) as (
    values (0), (1), (2), (3), (4), (5), (6), (7), (8), (9)
),

identifiers as (
    select n as identifier
    from base
    where n <= 3
)

select identifier as id
from identifiers;
