CREATE TABLE IF NOT EXISTS candidates (profile_id CHAR(11) PRIMARY KEY, first_name VARCHAR(32), last_name VARCHAR(32), country VARCHAR(16));

CREATE TABLE IF NOT EXISTS company (company_id CHAR(11) PRIMARY KEY,company_name VARCHAR(64),industry VARCHAR(64));

CREATE TABLE IF NOT EXISTS jobs (job_id CHAR(10) PRIMARY KEY, company_id CHAR(11) REFERENCES company (company_id), job_title VARCHAR(64), job_function VARCHAR(64), country VARCHAR(16), vacancies INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS application (app_id CHAR(11) PRIMARY KEY, job_id CHAR(11) REFERENCES jobs (job_id), profile_id CHAR(11) REFERENCES candidates (profile_id), datetime TIMESTAMP);

CREATE TABLE IF NOT EXISTS hired (app_id CHAR(11) REFERENCES application (app_id), datetime TIMESTAMP, hired BOOLEAN NOT NULL);

