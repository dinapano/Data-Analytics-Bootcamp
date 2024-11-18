--In this MySQL challenge, the table provided shows all new users signing up on a specific date in the format 
--YYYY-MM-DD.
--Your query should output the change from one month to the next. Because the first month has no preceding 
--month, your output should skip that row.

SELECT MONTHNAME(ULogin1.DateJoined) As Month,
COUNT(ULogin1.DateJoined) -
(
   SELECT COUNT(ULogin2.DateJoined)
   FROM maintable_862S4 AS ULogin2
   WHERE MONTH(ULogin2.DateJoined) = MONTH(ULogin1.DateJoined) - 1
) AS MonthToMonthChange
FROM maintable_862S4 AS ULogin1
WHERE MONTH(DateJoined) != 1
GROUP BY MONTHNAME(DateJoined)
ORDER BY DateJoined ASC;