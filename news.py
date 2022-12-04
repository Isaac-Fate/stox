"""
Implements functions to automatically scrape news headlines, and then insert them into the SQLite database. And it contains fetch news headlines from the database.
"""



import requests
from bs4 import BeautifulSoup, Tag
from datetime import datetime
from typing import Union
import os
import sqlite3
import pandas as pd

GOOGLE = "https://www.google.com"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:106.0) Gecko/20100101 Firefox/106.0"

def create_search_url(query: str, date: str) -> str:
    """Create a Google search URL based on provided query content and date.

    Parameters
    ----------
        query (str): The text typed into the search bar.
        date (str): Date string with format "YYYY-mm-dd".

    Returns
    -------
        str: Search URL.
    """
    
    # base URL
    url = GOOGLE
    
    # search content
    url += f"/search?q={query}"
    
    # search for news
    url += f"&tbm=nws"
    
    # get correct date format and then search by date
    date = datetime.strptime(date, "%Y-%m-%d")
    query_date = datetime.strftime(date, "%m/%d/%Y")
    url += f"&tbs=cdr:1,cd_min:{query_date},cd_max:{query_date}"
    
    # sort by relevancy
    url += f",sbd:0"
    
    # we want results in english
    url += "&lr=lang_en"
    
    return url

def find_links(url: str) -> list[str]:
    """Find links to news webpages.

    Parameters
    ----------
        url (str): News search URL.

    Returns
    -------
        list[str]: A list of links.
    """
    
    # prepare headers
    headers = {
        "User-Agent": USER_AGENT
    }

    # make a request
    res = requests.get(url, headers=headers)
    assert res.ok == True
    
    # make soup
    soup = BeautifulSoup(res.content, "html.parser")
    
    # find search tag
    search_tag = soup.find(id="search")

    # find tags containing links to webpages
    tags = search_tag.find_all(attrs={"class": "SoaBEf"})

    # extract links
    links = []
    tag: Tag
    for tag in tags:
        inner_tag: Tag = tag.find("a")
        link = inner_tag.get("href")
        links.append(link)
        
    return links

def extract_headline(link: str) -> str:
    """Extract the headline of a webpage given its link.

    Parameters
    ----------
        link (str): Link/URL to the webpage.

    Returns
    -------
        str: News headline.
    """
    
    # prepare headers
    headers = {
        "User-Agent": USER_AGENT
    }

    # make a request
    res = requests.get(link, headers=headers)
    assert res.ok == True
    
    # make soup
    soup = BeautifulSoup(res.content, "html.parser")
    
    # find headline
    headline_tag: Tag = soup.find("h1")
    headline = headline_tag.text.strip()
    
    return headline

def search_news_headline(query: str, date: str, skip: int = 0) -> Union[str, None]:
    """Find the news headline based on a query and sprecified date.

    Parameters
    ----------
        query (str): The text typed into the search bar.
        date (str): Date with format "YYYY-mm-dd".
        skip (int): Number of links to skip. Defaults to 0.

    Returns
    -------
        Union[str, None]: Returns a news headline if one is found.
    """
    
    url = create_search_url(query, date)
    links = find_links(url)
    
    # if no headline is found in some link,
    # then try the next one
    headline = None
    for link in links[skip:]:
        try:
            headline = extract_headline(link)
            break
        except:
            continue
    
    return headline

def init_db(db: os.PathLike, company: str):
    
    # connect to database
    con = sqlite3.connect(db)
    
    # create cursor
    cur = con.cursor()
    
    # initialize a table for the company
    cur.execute(
        f"""CREATE TABLE IF NOT EXISTS {company} (
            Date     TEXT PRIMARY KEY NOT NULL,
            Headline TEXT
        )
        """
    )
    
    # commit!
    con.commit()
    
    # close connection
    con.close()

def fetch_news_headline(db: os.PathLike, company: str, date: str) -> Union[str, None]:
    """Fetch one news headline from the data base given the company ticker and the date.

    Parameters
    ----------
        db (str): File path to data base.
        company (str): Company ticker.
        date (str): Date with format YYYY-mm-dd.

    Returns
    -------
        Union[str, None]: Returns a news headline if there is one.
    """
    
    # initialize database
    init_db(db, company)
    
    # connect to data base
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    # query
    res = cur.execute(
        f"""SELECT * FROM {company}
        WHERE Date == "{date}"
        """
    )
    row = res.fetchone() 
    
    # close data base
    con.commit()
    con.close()
    
    if row is not None:
        headline = row[1]
        if headline is None:
            headline = ""
        return headline
    else:
        return None

def fetch_news_headlines(db: os.PathLike, company: str) -> pd.DataFrame:
    """Fetch all news headlines from the data base given the company ticker.

    Parameters
    ----------
        db (str): File path to data base.
        company (str): Company ticker.

    Returns
    -------
        pd.DataFrame: A data frame consisting of one column named Headline, 
        and it is indexed by dates.
    """
    
    # initialize database
    init_db(db, company)
    
    # connect to data base
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    # query
    res = cur.execute(
        f"""SELECT * FROM {company}
        """
    )
    rows = res.fetchall() 
    
    # close data base
    con.commit()
    con.close()
    
    # convert rows to a data frame
    df = pd.DataFrame(rows, columns=["Date", "Headline"]).set_index("Date")
    df.index = pd.DatetimeIndex(df.index)
    
    return df

def search_and_insert_news_healine_to_db(
        db: os.PathLike,
        company: str,
        query: str,
        date: str,
        skip: int = 0
    ) -> str:
    
    # initialize database
    init_db(db, company)
    
    # connect to data base
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    # convert date string to right format
    date = datetime.strptime(date, "%Y-%m-%d")
    date = datetime.strftime(date, "%Y-%m-%d")
    
    # search news headline
    headline = search_news_headline(query, date, skip)
    
    # there exists a headline, we need to remove it
    if fetch_news_headline(db, company, date) is not None:
        cur.execute(
            f"""DELETE FROM {company}
            WHERE Date == "{date}"
            """
        )
    
    # insert news headline to database
    cur.execute(
        f"INSERT INTO {company} VALUES (?, ?)",
        (date, headline)
    )
    
    # close data base
    con.commit()
    con.close()
    
    return headline