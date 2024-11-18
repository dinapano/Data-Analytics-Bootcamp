--Retrieve the member firstname and surname who have made bookings for more than one facility using a SubQuery.

SELECT firstname, surname
FROM members
WHERE memid IN (SELECT memid FROM bookings GROUP BY memid HAVING COUNT(DISTINCT facid) > 1);