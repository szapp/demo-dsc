-- This SQL determines the identifiers that other SQLs are based on
with
base (n) as (values (0), (1), (2), (3), (4), (5), (6), (7), (8), (9)),

numbers as (
    select bsa.n + 10 * bsb.n as n
    from base as bsa cross join base as bsb
),

dates as (
    select date(:date_start, '+' || n || ' days') as 'date'
    from numbers
    where date(:date_start, '+' || n || ' days') <= :date_end
),

identifiers as (
    select n as identifier
    from base
    where n <= 3
)

select
    idn.identifier as id,
    dte.date
from identifiers as idn
cross join dates as dte
order by idn.identifier, dte.date;
