--In this MySQL challenge, your query should return all the people who report to either Jenny Richards or 
--have a NULL value in ReportsTo.
--The rows should be ordered by Age. Your query should also add a column at the end with a title of Boss 
--Title where the value is either None or CEO.

SELECT FirstName,
       LastName,
       ReportsTo,
       Position,
       Age,
       CASE
           WHEN ReportsTo = "Jenny Richards"
           THEN "CEO"
           ELSE "None"
       END AS "Boss Title"
FROM maintable_XNMIV
WHERE ReportsTo = "Jenny Richards"
   OR ReportsTo IS NULL
ORDER BY Age;