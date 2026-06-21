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
    -- E.g. missing identifiers from the index require data cleaning
    where n <= 3
)

select
    idn.identifier as id,
    dte.date,
    dte.counter * 3.1 as col1,
    dte.counter * 5.0 as col2,
    case dte.counter % 3
        when 0 then null
        else true
    end as col3,
    case dte.counter % 5
        when 0 then 'Apple'
        when 1 then 'Banana'
        when 2 then 'Cherry'
        when 3 then 'Date'
    end as col4
from identifiers as idn
cross join dates as dte
order by idn.identifier, dte.date;
