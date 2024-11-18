--In this MySQL challenge, your query should return the number of rows from your table where FirstName 
--contains "e" and LastName has more than 5 characters.

SELECT COUNT(*)
FROM maintable_AFMZP
WHERE FirstName LIKE "%e%" AND LastName LIKE "______%";