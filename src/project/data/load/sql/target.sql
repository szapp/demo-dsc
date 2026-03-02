WITH
base(n) AS ( VALUES (0), (1), (2), (3), (4), (5), (6), (7), (8), (9) ),

numbers AS (
    SELECT a.n + 10*b.n AS n
    FROM base a CROSS JOIN base b
),

dates AS (
    SELECT
        date(:start_date, '+' || n || ' days') AS date,
        n AS counter
    FROM numbers
    WHERE date(:start_date, '+' || n || ' days') <= :end_date
),

identifiers AS (
    SELECT n as identifier
    FROM base
    WHERE n <= 5 -- E.g. has more identifiers than the index
)

SELECT
    identifier AS id,
    date,
    counter as target
FROM identifiers
CROSS JOIN dates
ORDER BY identifier, date;
