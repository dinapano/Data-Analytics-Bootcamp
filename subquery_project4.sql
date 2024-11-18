--Retrieve the facility names along with the total number of bookings made for each facility 
--AS total_bookings, while using a SubQuery.

SELECT name, 
(SELECT COUNT(*) 
FROM bookings b
WHERE b.facid = f.facid) AS total_bookings
FROM facilities f;