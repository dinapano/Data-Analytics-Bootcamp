--Retrieve the facility names AS facility_name and the number of distinct members who have made bookings for 
--each facility in the "reatcodeltd_axldp_facilities," "reatcodeltd_axldp_bookings," and 
--"reatcodeltd_axldp_members" tables AS num_distinct_members, ordered by the number of distinct members 
--in descending order.

SELECT f.name AS facility_name, COUNT(DISTINCT b.memid) AS num_distinct_members
FROM facilities f
JOIN bookings b ON f.facid = b.facid
GROUP BY f.facid, f.name
ORDER BY num_distinct_members DESC;