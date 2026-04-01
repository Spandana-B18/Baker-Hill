"""
financial_spreading package

Sits between Azure Content Understanding extraction (step 2) and
normalization (step 3).  Takes the raw CU JSON, extracts financial
table rows, maps them to the Chart-of-Accounts schema, and returns a
structured spread JSON ready for display / download.
"""
