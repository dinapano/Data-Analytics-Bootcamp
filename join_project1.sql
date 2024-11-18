--Retrieve the firstname and the surname of all members who have made a booking in the "reatcodeltd_axldp_members"
-- and "reatcodeltd_axldp_bookings" tables.

SELECT DISTINCT
    m.firstname,
    m.surname
FROM members AS m
INNER JOIN bookings AS b ON m.memid = b.memid;