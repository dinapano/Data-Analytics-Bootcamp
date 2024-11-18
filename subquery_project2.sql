--Retrieve the facility names and their initialoutlay (from the reatcodeltd_axldp_facilities table) 
--where the initial outlay is equal to the maximum initial outlay across all facilities while using a SubQuery.

FROM facilities
WHERE initialoutlay = (SELECT MAX(initialoutlay) FROM facilities);