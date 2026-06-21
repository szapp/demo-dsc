with
base (n) as (values (0), (1), (2), (3), (4), (5), (6), (7), (8), (9)),

numbers as (
    select bsa.n + 10 * bsb.n as n
    from base as bsa cross join base as bsb
),

dates as (
    select
        n as counter,
        date(:start_date, '+' || n || ' days') as 'date'
    from numbers
    where date(:start_date, '+' || n || ' days') <= :end_date
),

identifiers as (
    select n as identifier
    from base
    where n <= 5 -- E.g. has more identifiers than the index
)

select
    idn.identifier as id,
    dte.date,
    dte.counter as target
from identifiers as idn
cross join dates as dte
order by idn.identifier, dte.date;
