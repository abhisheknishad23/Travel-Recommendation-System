# AI Travel Recommendation System & Popularity Predictor

An elegant, modern AI-powered travel dashboard built with Flask and Machine Learning. The system analyzes user history and preferences to recommend the top 5 destination matches using Cosine Similarity, while simultaneously predicting live popularity scores for any selected destination using a trained `RandomForestRegressor`.

##  Features
Dynamic Diagnostics Banner: Displays individual User ID along with the live predicted popularity score for the chosen destination.
Top 5 Smart Alternates: Generates 5 collaborative travel recommendations side-by-side in horizontal layout cards with metadata badges.
Fully Dynamic Inputs: Automatically fetches unique values for Destinations, States, Types, Best Time to Visit, and User Preferences from the underlying datasets into sleek drop-down menus.


---

## Project Structure
```text
my_travel_app/

app.py                           # Flask Backend Pipeline
travel_recommend.pkl             # Trained RandomForestRegressor Model
travel_encoders.pkl              # Trained Label Encoders
.gitignore                       # Git ignored tracking files
README.md                        # Documentation File
templates/
 index.html                   