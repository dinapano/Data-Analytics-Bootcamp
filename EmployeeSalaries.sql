--In this MySQL challenge, your query should return the information for the employee with the third highest 
--salary.
--Write a query that will find this employee and return that row, but then replace the DivisionID column with 
--the corresponding DivisionName from the table cb_companydivisions.
--You should also replace the ManagerID column with the ManagerName if the ID exists in the table and is not 
--NULL.

SELECT main.ID, 
       Name,
       DivisionName,
       CASE
           WHEN ManagerID IS NOT Null
           THEN (
               SELECT m1.Name
               FROM maintable_DV5BD m1
               WHERE m1.id = main.ManagerID
           )
           ELSE ManagerID
       END AS ManagerName,
       Salary
FROM maintable_DV5BD main
JOIN cb_companydivisions company
ON main.DivisionID = company.id
ORDER BY Salary DESC 
LIMIT 1
OFFSET 2;