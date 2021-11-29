SELECT c.company_name,t1.availNum,t2.hiredNum, (t1.availNum-t2.hiredNum) as openNum
FROM company c 
LEFT JOIN (SELECT c.company_id,SUM(j.vacancies) AS availNum
			FROM company c
			LEFT JOIN jobs j 
				ON j.company_id = c.company_id
			GROUP BY c.company_id
            ) t1 ON t1.company_id = c.company_id

LEFT JOIN (SELECT c.company_id,SUM(h.hired) AS hiredNum
			FROM company c
			LEFT JOIN jobs j 
				ON j.company_id = c.company_id
			LEFT JOIN application a 
				ON a.job_id = j.job_id
			LEFT JOIN hired h 
				ON h.app_id = a.app_id 
			GROUP BY c.company_id
            ) t2 ON t2.company_id = c.company_id