--In this MySQL challenge, your query should return FirstName, LastName, and the Age from your table 
--for all users who are above the age of 25 ordered by ID in ascending order.

SELECT FirstName, LastName, Age
FROM maintable_QCKCM
WHERE Age > 25