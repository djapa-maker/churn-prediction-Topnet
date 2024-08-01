from flask import Flask, request, render_template, url_for, redirect
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from utils.preprocess import preprocess_data
import sqlite3
import plotly.express as px
import plotly.io as pio
app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load('model/knn_model.pkl')
scaler = joblib.load('model/scaler.pkl')

DATABASE = 'app_database.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/home', methods=['GET'])
def home():
    try:
        conn = get_db_connection()
        query_client_data = 'SELECT * FROM client_data'
        query_facture_data = 'SELECT * FROM facture_data'
        client_data = pd.read_sql(query_client_data, conn)
        facture_data = pd.read_sql(query_facture_data, conn)
        conn.close()
        if client_data.empty or facture_data.empty:
            return render_template('index.html', message="No data available to display.")
        final_data = preprocess_data(client_data, facture_data)

        # Drop the target column if present
        X = final_data.drop(columns=['resiliation'], errors='ignore')
        X_scaled = scaler.transform(X)

        predictions = model.predict(X_scaled)

        # Ensure the predictions and the client_data lengths match
        if len(predictions) != len(client_data):
            client_data = client_data.iloc[:len(predictions)]

        # Add predictions to the original client data
        client_data['prediction'] = predictions

        # Churn Prediction Plot
        churn_counts = client_data['prediction'].value_counts().reset_index()
        churn_counts.columns = ['Prediction', 'Count']
        fig1 = px.bar(churn_counts, x='Prediction', y='Count', 
                     title='Churn Predictions Distribution',
                     color='Prediction', 
                     color_discrete_sequence=px.colors.qualitative.Set2)

        # Customize layout for fig1
        fig1.update_layout(
            xaxis_title='Prediction',
            yaxis_title='Count',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(255,255,255,0.8)',
            title_font_size=24,
            title_x=0.5
        )

        # Another Example Plot (e.g., Feature Distribution)
        churn_per_governorate = client_data.groupby('gouvernorat')['prediction'].mean().reset_index()
        churn_per_governorate.columns = ['Governorate', 'Average Churn Rate']
        fig2 = px.bar(churn_per_governorate, x='Governorate', y='Average Churn Rate',
                     title='Churn Rate per Governorate',
                     color='Average Churn Rate', 
                     color_continuous_scale=px.colors.sequential.Plasma)

        # Customize layout for fig2
        fig2.update_layout(
            xaxis_title='Governorate',
            yaxis_title='Average Churn Rate',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(255,255,255,0.8)',
            title_font_size=24,
            title_x=0.5
        )

        # Convert plots to HTML
        plot1_html = pio.to_html(fig1, full_html=False)
        plot2_html = pio.to_html(fig2, full_html=False)

        return render_template('index.html', plot1=plot1_html, plot2=plot2_html)

    
    except Exception as e:
        return f"Error processing data: {e}", 400
    
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file1 = request.files.get('datafile')
        file2 = request.files.get('datafacturefile')
        
        if file1 and file2:
            try:
                # Read CSV files
                data = pd.read_csv(file1)
                datafacture = pd.read_csv(file2)

                # Save CSV files data to the database
                conn = get_db_connection()
                data.to_sql('client_data', conn, if_exists='replace', index=False)
                datafacture.to_sql('facture_data', conn, if_exists='replace', index=False)
                conn.close()

                return redirect(url_for('home'))
            
            except Exception as e:
                return f"Error processing files: {e}", 400
    
    return render_template('upload.html')


@app.route('/', methods=['GET'])
def index():
    try:
        # Get the current page number and page size
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))

        # Process data from the database
        conn = get_db_connection()
        query_client_data = 'SELECT * FROM client_data'
        query_facture_data = 'SELECT * FROM facture_data'
        client_data = pd.read_sql(query_client_data, conn)
        facture_data = pd.read_sql(query_facture_data, conn)
        conn.close()

        final_data = preprocess_data(client_data, facture_data)

        # Drop the target column if present
        X = final_data.drop(columns=['resiliation'], errors='ignore')
        X_scaled = scaler.transform(X)

        predictions = model.predict(X_scaled)

        # Ensure the predictions and the client_data lengths match
        if len(predictions) != len(client_data):
            client_data = client_data.iloc[:len(predictions)]

        # Add predictions to the original client data
        client_data['prediction'] = predictions

        # Drop unwanted columns
        columns_to_drop = ['daysrc', 'governorat_mapped', 'type_abonnement','adresse','gouvernorat','delegation','Code postal']
        client_data = client_data.drop(columns=columns_to_drop, errors='ignore')

        # Add "See Facture" button column
        client_data['See Facture'] = client_data['new_codeclient'].apply(lambda x: f'<a href="/view_facture/{x}" class="btn btn-info">See Facture</a>')

        # Implement pagination
        total = len(client_data)
        pages = (total + per_page - 1) // per_page
        start = (page - 1) * per_page
        end = start + per_page
        paginated_data = client_data.iloc[start:end]

        # Convert the DataFrame to HTML and clean up newlines
        result_html = paginated_data.to_html(classes='data', index=False, escape=False)
        result_html = result_html.replace('\n', '').strip()

        return render_template('result.html', 
                               tables=result_html, 
                               titles=client_data.columns.values,
                               page=page,
                               pages=pages,
                               per_page=per_page)
    except Exception as e:
        return f"Error processing data: {e}", 400



@app.route('/view_facture/<client_id>')
def view_facture(client_id):
    try:
        conn = get_db_connection()
        query_facture = 'SELECT * FROM facture_data WHERE unique_codesclient = ?'
        facture_data = pd.read_sql(query_facture, conn, params=(client_id,))
        conn.close()

        if facture_data.empty:
            return "No facture found for this client.", 404

        # Convert the DataFrame to HTML and clean up newlines
        result_html = facture_data.to_html(classes='data', index=False)
        result_html = result_html.replace('\n', '').strip()

        return render_template('facture.html', tables=result_html, titles=facture_data.columns.values)
    except Exception as e:
        return f"Error processing facture: {e}", 400


if __name__ == '__main__':
    app.run(debug=True)
