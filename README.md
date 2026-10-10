# Terra — Fertilizer Recommendation Website

A Streamlit website for a dataset-based fertilizer-label prediction model.

## Files
- `app.py` — website
- `fertilizer_model.pkl` — export this from your trained Google Colab notebook
- `requirements.txt` — Python dependencies

## Run locally in PyCharm
1. Put `app.py`, `requirements.txt`, and `fertilizer_model.pkl` in the same project folder.
2. Open the terminal in that folder.
3. Create/activate a virtual environment if desired.
4. Run `pip install -r requirements.txt`
5. Run `streamlit run app.py`

## Deploy on Render
Create a **Web Service** connected to your GitHub repository.
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

Commit `fertilizer_model.pkl` only if you are comfortable storing the trained model in the repository. If you prefer not to store it in Git, use a private artifact download/storage flow and adapt `app.py` to load it securely.

## Important
The model predicts fertilizer labels from the dataset. Its output is not validated agronomic advice; verify fertilizer choice and application rates with soil testing and local agricultural guidance.
