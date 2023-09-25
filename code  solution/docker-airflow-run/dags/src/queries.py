CREATE_TABLE_ML_JOB = """
create table if not exists mljob (
    id serial primary key,
    job_id varchar(36) not null,
    job_type varchar(36),
    job_date date,
    stage varchar(36) not null,
    status varchar(36) not null,
    message text not null,
    created_at timestamp not null default now()
)
"""

CREATE_TEMP_TABLE_LOAN = """
    create temp table loan as (
        select 
            lower(t1.loan_id) loan_id,
            lower(t1.customer_id) customer_id,
            lower(t1.loan_status) loan_status,
            cast(concat(split_part(t1.application_time, '-', 2), '-', split_part(t1.application_time, '-', 1), '-', split_part(t1.application_time, '-', 3)) as timestamp) application_time,
            t1.current_loan_amount,
            lower(t1.term) term,
            t1.tax_liens,
            lower(t1.purpose) purpose,
            t1.no_of_properties 
        from (
            select 
                row_number() over(partition by loan_id order by application_time desc) rnk,
                ld.*
            from (
                select distinct * 
                from loan_details 
                where cast(concat(split_part(application_time, '-', 2), '-', split_part(application_time, '-', 1), '-', split_part(application_time, '-', 3)) as timestamp) between '{start_date}' and '{end_date}') ld
        ) t1
        where rnk=1
    );
"""

CREATE_TEMP_TABLE_CUSTOMER = """
    create temp table customer as (
        select t2.* 
        from (
            select customer_id, count(*) cnt from (select distinct * from customer_details) cd 
            group by customer_id 
        ) t1 
        join (select distinct * from customer_details) t2
        on t2.customer_id = t1.customer_id
        where t1.cnt=1
    );
"""

CREATE_TEMP_TABLE_CREDIT = """
    create temp table credit as (
        select t2.* 
        from (
            select customer_id, count(*) cnt from (select distinct * from credit_details) cd 
            group by customer_id 
        ) t1 
        join (select distinct * from credit_details) t2
        on t2.customer_id = t1.customer_id
        where t1.cnt=1
    );
"""

GET_DATA = """
    select 
        t1.loan_id, t1.customer_id, t1.loan_status, t1.application_time, t1.current_loan_amount, t1.term, t1.tax_liens, t1.purpose, t1.no_of_properties,
        lower(t2.home_ownership) home_ownership, t2.annual_income, lower(t2.years_in_current_job) years_in_current_job, t2.months_since_last_delinquent, t2.no_of_cars, t2.no_of_children,
        t3.credit_score, t3.monthly_debt, t3.years_of_credit_history, t3.no_of_open_accounts, t3.no_of_credit_problems, t3.current_credit_balance, t3.max_open_credit, t3.bankruptcies
    from loan t1
    left join customer t2
    on t2.customer_id = t1.customer_id
    left join credit t3
    on t3.customer_id = t2.customer_id
"""


LOG_ACTIVITY = """
    insert into mljob (
        job_id,
        job_type,
        job_date,
        stage,
        status,
        message
    ) values ('{job_id}', '{job_type}', '{job_date}', '{stage}', '{status}', '{message}')
"""

GET_JOB_STATUS = """
    select job_id, cast(job_date as varchar), stage, status, message, created_at 
    from mljob 
    where job_id = '{job_id}'
    order by created_at desc
    limit 1
"""

GET_JOB_DATE = """
    select job_date from mljob where job_id = '{job_id}'
    order by created_at desc
    limit 1
"""

GET_JOB_LOGS = """
    select job_id, cast(created_at as varchar), stage, status, message, created_at 
    from mljob 
    where job_id = '{job_id}' and message != '' and message is not null
    order by created_at desc
"""

GET_LATEST_TRAINING_JOB_ID = """
    select job_id from mljob
    where status = '{status}' and job_type = 'training' and stage = 'training'
    order by created_at desc
    limit 1
"""

GET_LATEST_DEPLOYED_JOB_ID = """
    select job_id from mljob
    where status = '{status}' and job_type = 'training' and stage = 'deploy'
    order by created_at desc
    limit 1
"""

GET_LATEST_JOB_ID = """
    select job_id from mljob
    where status = '{status}' and job_type = '{job_type}' and stage = '{stage}'
    order by created_at desc
    limit 1
"""