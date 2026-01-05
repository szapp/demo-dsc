WITH
base(n) AS ( VALUES (0),(1),(2),(3),(4),(5),(6),(7),(8),(9) ),
numbers(n) AS (
    SELECT a.n + 10 * b.n AS n
    FROM base a
    CROSS JOIN base b
)
SELECT
    date(:start_date, '+' || n || ' days') AS date,
    0.0 AS col1,
    n * 5.0 AS col2,
    n * 3.1 AS col3
FROM numbers
WHERE date(:start_date, '+' || n || ' days') <= :end_date
ORDER BY date;
