import requests
from bs4 import BeautifulSoup, Tag
from datetime import datetime
from typing import Union

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

def find_news_headline(query: str, date: str, skip: int = 0) -> Union[str, None]:
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
