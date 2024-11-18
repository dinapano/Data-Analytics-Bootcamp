--Retrieve the facility names along with the total number of bookings made for each facility in the 
--"reatcodeltd_axldp_facilities" and "reatcodeltd_axldp_bookings" tables as total_bookings.

SELECT DISTINCT
    m.firstname,
    m.surname
FROM members AS m
INNER JOIN bookings AS b ON m.memid = b.memid;