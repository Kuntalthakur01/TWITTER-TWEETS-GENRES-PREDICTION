from flask import Flask, jsonify, request, render_template
import pickle
import torch 

app = Flask(__name__)

# Load the model and tokenizer from the pickle file
with open("tweet_topic_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]

category_labels = {
    0: "arts_&_culture",
    1: "business_&_entrepreneurs",
    2: "celebrity_&_pop_culture",
    3: "diaries_&_daily_life",
    4: "family",
    5: "fashion_&_style",
    6: "film_tv_&_video",
    7: "fitness_&_health",
    8: "food_&_dining",
    9: "gaming",
    10: "learning_&_educational",
    11: "music",
    12: "news_&_social_concern",
    13: "other_hobbies",
    14: "relationships",
    15: "science_&_technology",
    16: "sports",
    17: "travel_&_adventure",
    18: "youth_&_student_life"
}

# Define a function to make predictions on new text
def predict_topic(text):
    encoded_text = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    logits = model(**encoded_text).logits
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    predicted_label = category_labels[predicted_class_idx]
    return predicted_label



# Define a route to handle POST requests to the server
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        tweet = request.form.get('tweet', '')
        if tweet:
            predicted_topic = predict_topic(tweet)
            return render_template('index.html', predicted_topic=predicted_topic)
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run()

