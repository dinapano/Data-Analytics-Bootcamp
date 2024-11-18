--In this MySQL challenge, your query should return LastName and the sum of Age from your table for all 
--users with a LastName = Smith. The column title of the summed ages should be SumAge.

SELECT LastName, SUM(Age) AS SumAge
FROM maintable_J8IA4
WHERE LastName = "Smith"
GROUP BY LastName;