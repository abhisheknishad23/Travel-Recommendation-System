from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Model aur Encoders load karein
model = pickle.load(open("travel_recommend.pkl", "rb"))
label_encoders = pickle.load(open("travel_encoders.pkl", "rb"))

# Notebook ke dataframes ki tarah data load karein taaki exact dynamic dropdowns banein
try:
    destination_df = pd.read_csv('Expanded_Destinations.csv')
    user_df = pd.read_csv('Final_Updated_Expanded_Users.csv')
    
    # Dropdown ke liye saari unique values extract karein
    destinations_list = sorted(destination_df['Name'].unique())
    states_list = sorted(destination_df['State'].unique())
    types_list = sorted(destination_df['Type'].unique())
    times_list = sorted(destination_df['BestTimeToVisit'].unique())
    preferences_list = sorted(user_df['Preferences'].dropna().unique())
except Exception as e:
    # Fallback lists (Testing ke liye agar files read na ho payein)
    destinations_list = ['Taj Mahal', 'Goa Beaches', 'Jaipur City', 'Kerala Backwaters', 'Leh Ladakh', 'Shimla Hills']
    states_list = ['Uttar Pradesh', 'Goa', 'Rajasthan', 'Kerala', 'Jammu and Kashmir', 'Himachal Pradesh']
    types_list = ['Historical', 'Beach', 'City', 'Nature', 'Adventure']
    times_list = ['Nov-Feb', 'Nov-Mar', 'Oct-Mar', 'Sep-Mar', 'Apr-Jun', 'Mar-Jun']
    preferences_list = ['Beaches, Historical', 'Nature, Adventure', 'City, Historical']

features = ['Name_x', 'State', 'Type', 'BestTimeToVisit', 'Preferences', 'Gender', 'NumberOfAdults', 'NumberOfChildren']

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    user_popularity = None
    user_id = None
    selected_dest = None
    
    if request.method == "POST":
        user_id = request.form.get('UserID')
        selected_dest = request.form.get('Name_x')
        selected_type = request.form.get('Type')
        
        # User dynamic input dictionary
        user_input = {
            'Name_x': selected_dest,
            'Type': selected_type,
            'State': request.form.get('State'),
            'BestTimeToVisit': request.form.get('BestTimeToVisit'),
            'Preferences': request.form.get('Preferences'),
            'Gender': request.form.get('Gender'),
            'NumberOfAdults': int(request.form.get('NumberOfAdults')),
            'NumberOfChildren': int(request.form.get('NumberOfChildren'))
        }
        
        # --- 1. CURRENT USER SELECTION POPULARITY ---
        input_df = pd.DataFrame([user_input])
        for col in features:
            if col in label_encoders:
                le = label_encoders[col]
                try:
                    input_df[col] = le.transform(input_df[col])
                except:
                    input_df[col] = le.transform([le.classes_[0]])[0]
                    
        input_df = input_df[features]
        user_popularity = round(model.predict(input_df)[0], 2)
        
        # --- 2. TOP 5 COLLABORATIVE RECOMMENDATIONS ---
        # Content Type ke filter se top 5 destinations filter karna (Notebook logic simulator)
        try:
            matched_places = destination_df[destination_df['Type'].str.lower() == selected_type.lower()].head(5)
            if len(matched_places) < 5:
                extra_places = destination_df[destination_df['Type'].str.lower() != selected_type.lower()].head(5 - len(matched_places))
                matched_places = pd.concat([matched_places, extra_places])
        except:
            # Fallback mock dataframe agar dynamic check custom tables par crash ho
            matched_places = pd.DataFrame([
                {'Name': 'Taj Mahal', 'State': 'Uttar Pradesh', 'Type': 'Historical', 'BestTimeToVisit': 'Nov-Feb', 'Preferences': 'City, Historical'},
                {'Name': 'Goa Beaches', 'State': 'Goa', 'Type': 'Beach', 'BestTimeToVisit': 'Nov-Mar', 'Preferences': 'Beaches, Historical'},
                {'Name': 'Jaipur City', 'State': 'Rajasthan', 'Type': 'City', 'BestTimeToVisit': 'Oct-Mar', 'Preferences': 'City, Historical'},
                {'Name': 'Kerala Backwaters', 'State': 'Kerala', 'Type': 'Nature', 'BestTimeToVisit': 'Sep-Mar', 'Preferences': 'Nature, Adventure'},
                {'Name': 'Leh Ladakh', 'State': 'Jammu and Kashmir', 'Type': 'Adventure', 'BestTimeToVisit': 'Apr-Jun', 'Preferences': 'Nature, Adventure'}
            ])
            
        for _, row in matched_places.iterrows():
            rec_input = user_input.copy()
            rec_input['Name_x'] = row['Name']
            rec_input['State'] = row['State']
            rec_input['Type'] = row['Type']
            rec_input['BestTimeToVisit'] = row['BestTimeToVisit']
            rec_input['Preferences'] = row['Preferences'] if 'Preferences' in row else row['Type']
            
            rec_df = pd.DataFrame([rec_input])
            for col in features:
                if col in label_encoders:
                    le = label_encoders[col]
                    try:
                        rec_df[col] = le.transform(rec_df[col])
                    except:
                        rec_df[col] = le.transform([le.classes_[0]])[0]
            
            rec_df = rec_df[features]
            rec_pop = round(model.predict(rec_df)[0], 2)
            
            recommendations.append({
                'name': row['Name'],
                'state': row['State'],
                'type': row['Type'],
                'best_time': row['BestTimeToVisit'],
                'score': rec_pop
            })
            
    return render_template("index.html", 
                           destinations=destinations_list,
                           states=states_list,
                           types=types_list,
                           times=times_list,
                           preferences=preferences_list,
                           user_id=user_id, 
                           selected_dest=selected_dest, 
                           user_popularity=user_popularity, 
                           recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)