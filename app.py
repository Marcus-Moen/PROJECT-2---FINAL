import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# NLTK setup
nltk.data.path.append('C:\\Users\\marcu\\AppData\\Roaming\\nltk_data')
nltk.download('punkt', download_dir='C:\\Users\\marcu\\AppData\\Roaming\\nltk_data', force=True)
nltk.download('stopwords', download_dir='C:\\Users\\marcu\\AppData\\Roaming\\nltk_data')

# Load model, vectorizer, and label encoder
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Emoji mapping
emoji_map = {
    'positive': '😊',
    'neutral': '😐',
    'negative': '😞'
}

# Bootstrap app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Tweet Sentiment Classifier"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <title>Tweet Sentiment Analyzer</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap" rel="stylesheet">
        {%metas%}
        {%favicon%}
        {%css%}
    </head>
    <body style="font-family: 'Poppins', sans-serif;">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([

                    html.H1("Tweet Sentiment Analyzer", className='text-center text-primary mb-4', 
                            style={'fontWeight': '800', 'fontSize': '3rem'}),

                    dcc.Textarea(
                        id='input-tweet',
                        placeholder='Type or paste a tweet here...',
                        style={
                            'width': '100%', 'height': 120, 'padding': '1rem',
                            'borderRadius': '10px', 'border': '1px solid #ced4da',
                            'fontSize': '16px'
                        }
                    ),

                    html.Small("e.g., 'I love the new phone update!' or 'This service is terrible 😡'", 
                               className='text-muted'),
                    html.Br(),
                    
                    dbc.Button("Analyze Sentiment", id='submit-button', n_clicks=0, color='primary', className='mt-3'),

                    dcc.Loading(
                        id="loading",
                        type="circle",
                        children=html.Div(id='prediction-output')
                    )
                ])
            ], className="p-4 shadow-lg bg-dark text-light mt-5")  # Added top margin
        ], width=12, lg=8, className="mx-auto")
    ]),

    html.Footer("© 2025 Tweet Analyzer | All rights reserved", 
                className='text-center mt-5 text-muted')
], fluid=True)

# Callback - using State so it only triggers on button click
@app.callback(
    Output('prediction-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input-tweet', 'value')
)
def predict_sentiment(n_clicks, input_text):
    if not n_clicks:
        return ""

    try:
        if input_text:
            cleaned = clean_text(input_text)
            if len(cleaned.strip()) < 3:
                return dbc.Alert("⚠️ Input is too short after cleaning. Try a more complete sentence.", color="warning", className="mt-4")

            vector = vectorizer.transform([cleaned])
            if vector.nnz == 0:
                return dbc.Alert("⚠️ The input text doesn't contain recognizable words. Try a different tweet.", color="warning", className="mt-4")

            prediction = model.predict(vector)[0]
            sentiment_label = label_encoder.inverse_transform([prediction])[0]
            emoji = emoji_map.get(sentiment_label.lower(), '')
            return dbc.Alert(f"Predicted Sentiment: {sentiment_label.capitalize()} {emoji}", color="info", className="mt-4", style={'fontSize': '1.5rem'})

        return ""
    except Exception as e:
        import traceback
        traceback.print_exc()
        return dbc.Alert(f"❌ Error: {str(e)}", color="danger", className="mt-4")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)


