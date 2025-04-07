import pandas
from jobspy import scrape_jobs


def jobspy_scrape_jobs(
    site_names: list[str], terms: str, location: str, num_results: int, time_limit: int
) -> pandas.DataFrame:
    """
    Scrape jobs from job boards such as LinkedIn and glassdoor

    Args:
        site_names (list): List of job board sites to scrape from, available sites are linkedin and glassdoor
        terms (list): List of job titles to search for
        location (str): Location to search for jobs
        num_results (int): Number of results to return
        time_limit (int): Time limit in hours to scrape for jobs
    """
    return scrape_jobs(
        site_name=site_names,
        search_term=terms,
        location=location,
        results_wanted=num_results,
        hours_old=time_limit,
    )
