--In this MySQL challenge, your query should return the vendor information along with the values from the 
--table cb_vendorinformation.
--You should combine the values of the two tables based on the GroupID column. 
--The final query should only print out the GroupID, CompanyName, and final count of all rows that are 
--grouped into each company name under a column titled Count. The output table should be then sorted by the 
--Count column and then sorted by GroupID so that a higher number appears first.

SELECT m.GroupID, c.CompanyName, COUNT(*) as Count
FROM maintable_DFZDV AS m
JOIN cb_vendorinformation AS c ON m.GroupID = c.GroupID
GROUP BY m.GroupID
ORDER BY Count, m.GroupID DESC;