import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import plotly.graph_objs as go

# NLTK setup
nltk.data.path.append('C:\\Users\\marcu\\AppData\\Roaming\\nltk_data')
nltk.download('punkt', download_dir='C:\\Users\\marcu\\AppData\\Roaming\\nltk_data', force=True)
nltk.download('stopwords', download_dir='C:\\Users\\marcu\\AppData\\Roaming\\nltk_data')
nltk.download('wordnet', download_dir='C:\\Users\\marcu\\AppData\\Roaming\\nltk_data')
nltk.download('omw-1.4', download_dir='C:\\Users\\marcu\\AppData\\Roaming\\nltk_data') 

# Load model, vectorizer, and label encoder
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    try:
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'\@\w+|\#','', text)
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error cleaning text: {text} -> {e}")
        return ""

emoji_map = {
    'positive': '😊',
    'neutral': '😐',
    'negative': '😞'
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "JetSent - Airline Sentiment Analyzer"
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <title>JetSent - Airline Sentiment Analyzer</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap" rel="stylesheet">
        {%metas%}
        {%favicon%}
        {%css%}
        <style>
            .heading-brand {
                font-weight: 800;
                font-size: 3.5rem;
                color: #00d2ff;
                background: -webkit-linear-gradient(45deg, #00d2ff, #3a47d5);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .btn-primary {
                background-color: #007bff;
                border-color: #007bff;
                transition: background-color 0.3s ease, transform 0.2s ease;
            }

            .btn-primary:hover {
                background-color: #0056b3;
                transform: scale(1.05);
            }

            .btn:focus {
                outline: none;
                box-shadow: 0 0 0 0.2rem rgba(0,123,255,.5);
            }
        </style>
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

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([

            html.Div([
                html.H1("🛩️ JetSent", className='text-center heading-brand mb-2'),
                html.H5("AI-powered sentiment analysis for airline reviews", className='text-center text-muted mb-4')
            ]),

            dbc.Card([
                dbc.CardBody([

                    dcc.Textarea(
                        id='input-review',
                        placeholder='Type or paste a review here...',
                        style={
                            'width': '100%', 'height': 120, 'padding': '1rem',
                            'borderRadius': '10px', 'border': '1px solid #ced4da',
                            'fontSize': '16px'
                        }
                    ),

                    html.Small("e.g., 'I love the new flight experience!' or 'This airline is the worst 😡'", 
                               className='text-muted'),
                    html.Br(),

                    dbc.Button("Analyze Sentiment", id='submit-button', n_clicks=0, color='primary', className='mt-3'),

                    dcc.Loading(
                        id="loading",
                        type="circle",
                        children=html.Div(id='prediction-output')
                    )
                ])
            ], className="p-4 shadow-lg bg-dark text-light mt-4")

        ], width=12, lg=8, className="mx-auto")
    ], className="mt-5"),

    html.Footer("© 2025 JetSent | All rights reserved", 
                className='text-center mt-5 text-muted')
], fluid=True)

@app.callback(
    Output('prediction-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input-review', 'value')
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
                return dbc.Alert("⚠️ The input text doesn't contain recognizable words. Try a different review.", color="warning", className="mt-4")

            prediction = model.predict(vector)[0]
            sentiment_label = label_encoder.inverse_transform([prediction])[0]
            emoji = emoji_map.get(sentiment_label.lower(), '✈️')

            probs = model.predict_proba(vector)[0]
            class_labels = label_encoder.inverse_transform(model.classes_)

            # Plotly Donut Chart
            # Enhanced Donut Chart
            chart = dcc.Graph(
                figure=go.Figure(
                    data=[go.Pie(
                        labels=[label.capitalize() for label in class_labels],
                        values=probs,
                        hole=0.55,
                        marker=dict(
                            colors=['#00cc96', '#a1a1a1', '#ef553b'],  # green, gray, red
                            line=dict(color='#000000', width=2)
                        ),
                        textinfo='percent',
                        textfont_size=18,
                        hoverinfo='label+percent+value',
                        pull=[0.1 if label == sentiment_label else 0 for label in class_labels],
                        sort=False,
                        insidetextorientation='radial'
                    )],
                    layout=go.Layout(
                        title=dict(text='Sentiment Confidence', font_size=24, x=0.5),
                        annotations=[dict(text=sentiment_label.capitalize(), x=0.5, y=0.5, font_size=22, showarrow=False)],
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        showlegend=True,
                        legend=dict(orientation="h", y=-0.2)
                    )
                ),
                config={'displayModeBar': False}
            )


            return html.Div([
                dbc.Alert(f"Predicted Sentiment: {sentiment_label.capitalize()} {emoji}", color="info", className="mt-4", style={'fontSize': '1.5rem'}),
                chart
            ])

        return ""
    except Exception as e:
        import traceback
        traceback.print_exc()
        return dbc.Alert(f"❌ Error: {str(e)}", color="danger", className="mt-4")

if __name__ == '__main__':
    app.run(debug=True)
