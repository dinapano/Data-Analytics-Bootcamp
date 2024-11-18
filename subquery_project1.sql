--Retrieve the member firstname and surname who have made bookings for facility ID 1. Using a SubQuery.

SELECT firstname, surname
FROM members
WHERE memid IN (SELECT memid FROM bookings WHERE facid = 1);