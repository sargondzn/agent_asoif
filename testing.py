import requests
from langchain_core.tools import tool

API = "https://awoiaf.westeros.org/api.php"
HEADERS = {"User-Agent": "asoif-study-bot/0.1"}
MAX_CHARS = 4000


@tool
def search_wiki(query: str) -> str:
    """Search A Wiki of Ice and Fire for a topic and return the page text.

    Use this for any question about characters, houses, places, or events in
    A Song of Ice and Fire. Pass a short search term, ideally a page title
    such as "Jon Snow" or "House Stark".
    """
    try:
        r = requests.get(API, params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
        }, headers=HEADERS, timeout=20)
        r.raise_for_status()
        hits = r.json()["query"]["search"]

        if not hits:
            return f"No wiki page found for '{query}'."

        title = hits[0]["title"]

        r = requests.get(API, params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "format": "json",
        }, headers=HEADERS, timeout=20)
        r.raise_for_status()
        pages = r.json()["query"]["pages"]
        text = next(iter(pages.values())).get("extract", "")

        if not text.strip():
            return f"Page '{title}' exists but returned no text."

        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + "\n\n[truncated]"

        return f"Page: {title}\n\n{text}"

    except Exception as e:
        return f"Wiki lookup failed: {e}"


if __name__ == "__main__":
    print(search_wiki.invoke("Jon Snow")[:500])
