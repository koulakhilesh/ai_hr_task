SELECT t2.industry, t2.company_name,t6.local_candiates
FROM(
	SELECT DISTINCT(c.industry),c.company_name, COUNT(DISTINCT t1.job_title) as dis_job
	FROM company c 
	LEFT JOIN( SELECT j.company_id, j.job_title
				FROM jobs j
				)t1 on t1.company_id=c.company_id
	GROUP BY c.industry,c.company_name
	ORDER BY c.industry , dis_job DESC)t2 
CROSS JOIN (SELECT t4.industry, t4.company_name,MAX(t4.dis_job) as dis_job
			FROM(
			SELECT DISTINCT(c.industry),c.company_name, COUNT(DISTINCT t3.job_title) as dis_job
			FROM company c 
			LEFT JOIN( SELECT j.company_id, j.job_title
						FROM jobs j
						)t3 on t3.company_id=c.company_id
			GROUP BY c.industry,c.company_name
			ORDER BY c.industry , dis_job DESC)t4 GROUP BY t4.industry,t4.company_name )t5 ON t2.industry = t5.industry AND t2.dis_job > t5.dis_job   
    
LEFT JOIN (SELECT c.company_name,COUNT(cn.country) AS local_candiates
			FROM company c
			LEFT JOIN jobs j 
				ON j.company_id = c.company_id
			LEFT JOIN application a 
				ON a.job_id = j.job_id
			LEFT JOIN candidates cn 
				ON cn.profile_id = a.profile_id 
			WHERE cn.country='SGP'    
			GROUP BY c.company_id) t6 ON t6.company_name = t2.company_name     
WHERE t6.local_candiates IS NOT NULL           