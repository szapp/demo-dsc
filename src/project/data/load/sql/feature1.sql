WITH
base(n) AS ( VALUES (0), (1), (2), (3), (4), (5), (6), (7), (8), (9) ),

numbers AS (
    SELECT a.n + 10 * b.n AS n
    FROM base a CROSS JOIN base b
),

dates AS (
    SELECT
        date(:start_date, '+' || n || ' days') as date,
        n as counter
    FROM numbers
    WHERE date(:start_date, '+' || n || ' days') <= :end_date
),

identifiers AS (
    SELECT n as identifier
    FROM base
    WHERE n <= 3 -- E.g. if it misses some identifiers from the index, requires data cleaning
)

SELECT
    identifier AS id,
    date,
    0.0 AS col1,
    counter * 5.0 AS col2,
    counter * 3.1 AS col3
FROM identifiers
CROSS JOIN dates
ORDER BY identifier, date;
