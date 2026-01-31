# QUICK FIX - Run this in a new cell in your Jupyter notebook
# This will download the missing punkt_tab resource

import nltk
print("Downloading punkt_tab...")
nltk.download('punkt_tab')
print("✅ Done! Now restart kernel and run all cells.")
