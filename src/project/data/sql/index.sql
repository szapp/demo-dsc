-- This SQL determines the identifiers that other SQLs are based on
WITH
base(n) AS ( VALUES (0), (1), (2), (3), (4), (5), (6), (7), (8), (9) ),

numbers AS (
    SELECT a.n + 10*b.n AS n
    FROM base a CROSS JOIN base b
),

dates AS (
    SELECT date(:start_date, '+' || n || ' days') AS date
    FROM numbers
    WHERE date(:start_date, '+' || n || ' days') <= :end_date
),

identifiers AS (
    SELECT n AS identifier
    FROM base
    WHERE n <= 3
)

SELECT
    identifier AS id,
    date
FROM identifiers
CROSS JOIN dates
ORDER BY identifier, date;
