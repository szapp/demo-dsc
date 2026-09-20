-- This SQL determines ont of the identifiers that other SQLs are based on
with
base (n) as (
    values (0), (1), (2), (3), (4), (5), (6), (7), (8), (9)
),

numbers as (
    select bsa.n + 10 * bsb.n as n
    from base as bsa cross join base as bsb
),

dates as (
    select date(:date_start, '+' || n || ' days') as 'date'
    from numbers
    where date(:date_start, '+' || n || ' days') <= :date_end
)

select dte.date
from dates as dte;
