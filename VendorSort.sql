--In this MySQL challenge, your query should return the vendor information along with 
--the values from the table cb_vendorinformation.
--You should combine the values of the two tables based on the GroupID column.
--The final query should consolidate the rows to be grouped by FirstName, and a Count column should be 
--added at the end that adds up the number of times that person shows up in the table.
--The output table should be sorted by the Count column in ascending order and then sorted by CompanyName 
--in alphabetical order.

SELECT main.GroupID,FirstName,LastName,Job,ExternalID,CompanyName,Count(*)
AS 'Count'
FROM maintable_2XNQC AS main
INNER JOIN cb_vendorinformation AS vendor
ON vendor.GroupId = main.GroupID
GROUP BY FirstName
ORDER BY Count ASC, CompanyName ASC