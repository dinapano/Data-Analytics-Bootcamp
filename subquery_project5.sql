--Write a SQL query to find all members who have recommended more people than the average number
--of recommendations made by all members.
--The output should include the member's ID, first name, surname, and the number of people they have recommended.
--Order the results by the number of recommendations in descending order.

SELECT 
    m.memid,
    m.firstname,
    m.surname,
    COUNT(r.memid) AS Recommendations
FROM members m
JOIN members r ON m.memid = r.recommendedby
GROUP BY m.memid, m.firstname, m.surname
HAVING COUNT(r.memid) > 
(SELECT AVG(Recommendation_Count) 
FROM (SELECT COUNT(memid) 
AS Recommendation_Count 
FROM members
WHERE recommendedby is not null 
GROUP BY recommendedby) AS subquery) 
ORDER BY Recommendations DESC;