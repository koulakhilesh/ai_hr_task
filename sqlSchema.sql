
CREATE TABLE IF NOT EXISTS candidates (profile_id CHAR(11) PRIMARY KEY, first_name VARCHAR(32), last_name VARCHAR(32), country VARCHAR(16));
INSERT INTO candidates (profile_id, first_name, last_name, country) values (1, 'Kailee', 'Pitts', 'AUS');
INSERT INTO candidates (profile_id, first_name, last_name, country) values (2, 'Virginia', 'Bruce', 'BEL');
INSERT INTO candidates (profile_id, first_name, last_name, country) values (3, 'Hazel', 'Ross', 'BRA');
INSERT INTO candidates (profile_id, first_name, last_name, country) values (4, 'Clarissa', 'Underwood', 'CAN');
INSERT INTO candidates (profile_id, first_name, last_name, country) values (5, 'Nathanael', 'Gray', 'DEU');
INSERT INTO candidates (profile_id, first_name, last_name, country) values (6, 'Nancy', 'Ortega', 'GBR');
INSERT INTO candidates (profile_id, first_name, last_name, country) values (7, 'Kristen', 'Haynes', 'JPN');
INSERT INTO candidates (profile_id, first_name, last_name, country) values (8, 'Zachariah', 'Carlson', 'LKA');
INSERT INTO candidates (profile_id, first_name, last_name, country) values (9, 'Rhett', 'Johns', 'NLD');
INSERT INTO candidates (profile_id, first_name, last_name, country) values (10, 'Martha', 'Perkins', 'NZL');
INSERT INTO candidates (profile_id, first_name, last_name, country) values (11, 'Dorian', 'Swanson', 'PRT');
INSERT INTO candidates (profile_id, first_name, last_name, country) values (12, 'Dustin', 'Giles', 'SGP');



CREATE TABLE IF NOT EXISTS company (company_id CHAR(11) PRIMARY KEY,company_name VARCHAR(64),industry VARCHAR(64));
INSERT INTO company (company_id, company_name, industry) values (1, 'Energy_Stack', 'Renewable');
INSERT INTO company (company_id, company_name, industry) values (2, 'Better_Food', 'Food_and_beverage ');
INSERT INTO company (company_id, company_name, industry) values (3, 'Xeno_Semicon', 'Semiconductor');


CREATE TABLE IF NOT EXISTS jobs (job_id CHAR(10) PRIMARY KEY, company_id CHAR(11) REFERENCES company (company_id), job_title VARCHAR(64), job_function VARCHAR(64), country VARCHAR(16), vacancies INTEGER NOT NULL);
INSERT INTO jobs (job_id, company_id, job_title, job_function, country, vacancies) values (1, 1, 'Chief_Scientist', 'technical', 'CAN', 1);
INSERT INTO jobs (job_id, company_id, job_title, job_function, country, vacancies) values (2, 1, 'Project_Manager', 'admin', 'CAN', 2);
INSERT INTO jobs (job_id, company_id, job_title, job_function, country, vacancies) values (3, 1, 'Data_Scientist', 'technical', 'CAN', 3);

INSERT INTO jobs (job_id, company_id, job_title, job_function, country, vacancies) values (4, 2, 'QA_Executive', 'technical', 'BEL', 2);
INSERT INTO jobs (job_id, company_id, job_title, job_function, country, vacancies) values (5, 2, 'Food_Technologist', 'technical', 'BEL', 2);
INSERT INTO jobs (job_id, company_id, job_title, job_function, country, vacancies) values (6, 2, 'HR_Manager', 'admin', 'BEL', 1);
INSERT INTO jobs (job_id, company_id, job_title, job_function, country, vacancies) values (7, 3, 'Associate_Test_Engineer', 'technical', 'SGP', 3);
INSERT INTO jobs (job_id, company_id, job_title, job_function, country, vacancies) values (8, 3, 'Senior_Process_Engineer', 'technical', 'SGP', 2);
INSERT INTO jobs (job_id, company_id, job_title, job_function, country, vacancies) values (9, 3, 'Quality_Manager', 'admin', 'SGP', 2);



CREATE TABLE IF NOT EXISTS application (app_id CHAR(11) PRIMARY KEY, job_id CHAR(11) REFERENCES jobs (job_id), profile_id CHAR(11) REFERENCES candidates (profile_id), datetime TIMESTAMP);
INSERT INTO application (app_id, job_id, profile_id, datetime) values (1, 1, 1, '2021-4-11 10:00:00');
INSERT INTO application (app_id, job_id, profile_id, datetime) values (2, 2, 2, '2021-4-17 11:00:00');
INSERT INTO application (app_id, job_id, profile_id, datetime) values (3, 3, 3, '2021-4-19 12:00:00');
INSERT INTO application (app_id, job_id, profile_id, datetime) values (4, 3, 4, '2021-4-25 13:00:00');
INSERT INTO application (app_id, job_id, profile_id, datetime) values (5, 4, 5, '2021-6-10 14:00:00');
INSERT INTO application (app_id, job_id, profile_id, datetime) values (6, 4, 6, '2021-6-13 15:00:00');
INSERT INTO application (app_id, job_id, profile_id, datetime) values (7, 5, 7, '2021-6-29 15:00:00');
INSERT INTO application (app_id, job_id, profile_id, datetime) values (8, 6, 8, '2021-6-24 17:00:00');
INSERT INTO application (app_id, job_id, profile_id, datetime) values (9, 7, 9, '2021-5-28 18:00:00');
INSERT INTO application (app_id, job_id, profile_id, datetime) values (10, 7, 10, '2021-5-26 19:00:00');
INSERT INTO application (app_id, job_id, profile_id, datetime) values (11, 8, 11, '2021-5-18 20:00:00');
INSERT INTO application (app_id, job_id, profile_id, datetime) values (12, 9, 12, '2021-5-15 21:00:00');





CREATE TABLE IF NOT EXISTS hired (app_id CHAR(11) REFERENCES application (app_id), datetime TIMESTAMP, hired BOOLEAN NOT NULL);
INSERT INTO hired (app_id, datetime, hired) values (1, '2021-4-22 10:00:00', TRUE);
INSERT INTO hired (app_id, datetime, hired) values (2, '2021-4-23 11:00:00', TRUE);
INSERT INTO hired (app_id, datetime, hired) values (3, '2021-4-25 12:00:00', FALSE);
INSERT INTO hired (app_id, datetime, hired) values (4, '2021-4-30 14:00:00', TRUE);
INSERT INTO hired (app_id, datetime, hired) values (5, '2021-6-15 15:00:00', FALSE);
INSERT INTO hired (app_id, datetime, hired) values (6, '2021-6-17 10:00:00', FALSE);
INSERT INTO hired (app_id, datetime, hired) values (7, '2021-7-4 11:00:00', TRUE);
INSERT INTO hired (app_id, datetime, hired) values (8, '2021-7-10 12:00:00', TRUE);
INSERT INTO hired (app_id, datetime, hired) values (9, '2021-6-12 14:00:00', TRUE);
INSERT INTO hired (app_id, datetime, hired) values (10, '2021-6-14 15:00:00', FALSE);
INSERT INTO hired (app_id, datetime, hired) values (11, '2021-6-16 10:00:00', TRUE);
INSERT INTO hired (app_id, datetime, hired) values (12, '2021-6-19 11:00:00', FALSE);
	
