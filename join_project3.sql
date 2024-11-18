--Retrieve the facility names AS facility_name and the member firstname and surname who have made bookings for
-- those facilities in the "reatcodeltd_axldp_facilities," "reatcodeltd_axldp_bookings," and 
--"reatcodeltd_axldp_members" tables, ordered by facility name and member surname, with a limit of 10 records.

SELECT f.name AS facility_name, m.firstname, m.surname
FROM facilities f
JOIN bookings b ON f.facid = b.facid
JOIN members m ON m.memid = b.memid
ORDER BY f.name, m.surname
LIMIT 10;