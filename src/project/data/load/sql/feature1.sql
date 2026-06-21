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
    counter * 3.1 AS col1,
    counter * 5.0 AS col2,
    CASE counter % 3
        WHEN 0 THEN null
        ELSE true
    END AS col3,
    CASE counter % 5
        WHEN 0 THEN 'Apple'
        WHEN 1 THEN 'Banana'
        WHEN 2 THEN 'Cherry'
        WHEN 3 THEN 'Date'
        ELSE null
    END AS col4
FROM identifiers
CROSS JOIN dates
ORDER BY identifier, date;
