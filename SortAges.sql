--In this MySQL challenge, your query should return the rows from your table where LastName = Smith or 
--FirstName = Robert and the rows should be sorted by Age in ascending order.

SELECT * FROM maintable_KRDCN
WHERE LastName = "Smith" OR FirstName = "Robert"
ORDER BY Age;