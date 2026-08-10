"""
Ticker -> company-name aliases for GDELT news linkage.

GDELT has no per-ticker feed; articles are matched by searching for company
names in the headline/body. Each ticker maps to a list of search phrases.
Guidelines that keep linkage noise down:
  * Use distinctive full names ("Advanced Micro Devices"), not ambiguous
    short forms ("AMD" alone also matches unrelated uses).
  * Include the common brand ("iPhone maker" is too loose; "Apple Inc" plus
    a couple of unambiguous product/brand terms is fine).
  * AVOID single common words that collide with everyday language
    (e.g., "Visa" the card network vs. travel visas, "Meta" vs. metadata).
    For those, prefer the multi-word legal name and a safe brand term.
The aliases are OR-combined and quoted as exact phrases in the query, which
sharply reduces false matches compared with bare tickers.
"""

TICKER_ALIASES: dict[str, list[str]] = {
    "AAPL":  ['"Apple Inc"', '"Apple iPhone"', '"Apple stock"'],
    "MSFT":  ['"Microsoft"', '"Microsoft Azure"'],
    "GOOGL": ['"Google"', '"Alphabet Inc"'],
    "AMZN":  ['"Amazon.com"', '"Amazon stock"', '"Amazon Web Services"'],
    "META":  ['"Meta Platforms"', '"Facebook parent"', '"Mark Zuckerberg"'],
    "NVDA":  ['"Nvidia"', '"Nvidia stock"'],
    "TSLA":  ['"Tesla Inc"', '"Tesla stock"', '"Elon Musk Tesla"'],
    "JPM":   ['"JPMorgan"', '"JPMorgan Chase"'],
    "JNJ":   ['"Johnson & Johnson"'],
    "XOM":   ['"Exxon Mobil"', '"ExxonMobil"'],
    "WMT":   ['"Walmart"'],
    "PG":    ['"Procter & Gamble"'],
    "V":     ['"Visa Inc"', '"Visa card"', '"Visa stock"'],
    "UNH":   ['"UnitedHealth"', '"UnitedHealth Group"'],
    "HD":    ['"Home Depot"'],
    "DIS":   ['"Walt Disney"', '"Disney stock"', '"Disney+"'],
    "NFLX":  ['"Netflix"'],
    "AMD":   ['"Advanced Micro Devices"', '"AMD stock"', '"AMD Ryzen"'],
    "BA":    ['"Boeing"'],
    "PFE":   ['"Pfizer"'],
}
