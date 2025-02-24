from flask import Flask, render_template, request, jsonify
import pandas as pd
from googleapiclient.discovery import build
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
# import seaborn as sns
import numpy as np
from PIL import Image
# import time
# from threading import Thread
# from queue import Queue
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Use environment variables in your code
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

app = Flask(__name__)

# Your API key
# api_key = 'AIzaSyD9_6gPUvffWe7ZlZrCdkJsBSb4MfaYTdw'
api_key = YOUTUBE_API_KEY

# Global variables for tracking progress
progress_data = {
    'current_step': 0,
    'status': 'idle',
    'message': '',
    'error': None
}

def reset_progress():
    progress_data['current_step'] = 0
    progress_data['status'] = 'idle'
    progress_data['message'] = ''
    progress_data['error'] = None

def update_progress(step, message):
    progress_data['current_step'] = step
    progress_data['message'] = message
    progress_data['status'] = 'processing'

def extract_video_id(youtube_url):
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, youtube_url)
    if match:
        return match.group(1)
    return None

def video_comments(video_id):
    replies = []
    limit = 9000
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_response = youtube.commentThreads().list(part='snippet,replies', videoId=video_id).execute()

    while video_response and len(replies) < limit:
        for item in video_response['items']:
            if len(replies) >= limit:
                return replies[:limit]
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            replies.append(comment)

            if 'replies' in item:
                for reply in item['replies']['comments']:
                    if len(replies) >= limit:
                        return replies[:limit]
                    repl = reply['snippet']['textDisplay']
                    replies.append(repl)

        if 'nextPageToken' in video_response and len(replies) < limit:
            video_response = youtube.commentThreads().list(
                part='snippet,replies',
                pageToken=video_response['nextPageToken'],
                videoId=video_id
            ).execute()
        else:
            break
    return replies[:limit]
#PRE-PROCESSING
def get_stopwords():
    # Memuat daftar stopword dari file dan mengembalikan dalam bentuk set
    try:
        with open('stopwordbahasa.xls', 'r', encoding='utf-8') as file:
            stopwords = {line.strip() for line in file if line.strip()}
        return stopwords
    except Exception as e:
        print(f"Error membaca file stopwords: {e}")
        return set()

def load_kamus():
    # Memuat kamus kata dasar dari file files/kamuss.txt
    try:
        with open('kamuss.txt', 'r', encoding='utf-8') as file:
            return {line.strip() for line in file if line.strip()}
    except Exception as e:
        print(f"Error membaca kamus: {e}")
        return set()

def cek_kata_dasar(kata, kamus):
    # Mengecek apakah kata ada dalam kamus kata dasar
    return kata in kamus

def hapus_inflection_suffixes(kata):
    # Menghapus inflectional suffixes (-lah, -kah, -ku, -mu, atau -nya)
    if kata.endswith(('lah', 'kah')):
        return kata[:-3]
    if kata.endswith(('ku', 'mu')):
        return kata[:-2]
    if kata.endswith('nya'):
        return kata[:-3]
    if kata.endswith('tah'):
        return kata[:-3]
    if kata.endswith('pun'):
        return kata[:-3]
    return kata

def hapus_derivation_suffixes(kata):
    # Menghapus derivational suffixes (-i, -an, atau -kan)
    if kata.endswith('i'):
        return kata[:-1]
    if kata.endswith('an'):
        return kata[:-2]
    if kata.endswith('kan'):
        return kata[:-3]
    return kata

def hapus_derivation_prefix(kata, kamus):
    # Menghapus derivational prefixes (di-, ke-, se-, me-, be-, pe-, te-)
    if len(kata) <= 4:
        return kata

    # Cek awalan di- dan ke-
    if kata.startswith(('di', 'ke')) and len(kata) > 4:
        kata_tanpa_awalan = kata[2:]
        if cek_kata_dasar(kata_tanpa_awalan, kamus):
            return kata_tanpa_awalan

    # Cek awalan se-
    if kata.startswith('se'):
        kata_tanpa_awalan = kata[2:]
        if cek_kata_dasar(kata_tanpa_awalan, kamus):
            return kata_tanpa_awalan

    # Cek awalan me-
    if kata.startswith('me'):
        kata_tanpa_awalan = kata[2:]
        if kata.startswith('meng'):
            kata_tanpa_awalan = kata[4:]
        elif kata.startswith('meny'):
            kata_tanpa_awalan = 's' + kata[4:]
        elif kata.startswith('men'):
            kata_tanpa_awalan = kata[3:]
        elif kata.startswith('mem'):
            kata_tanpa_awalan = 'p' + kata[3:]
        if cek_kata_dasar(kata_tanpa_awalan, kamus):
            return kata_tanpa_awalan

    # Cek awalan be-
    if kata.startswith('be'):
        kata_tanpa_awalan = kata[2:]
        if kata.startswith('ber'):
            kata_tanpa_awalan = kata[3:]
        if cek_kata_dasar(kata_tanpa_awalan, kamus):
            return kata_tanpa_awalan

    # Cek awalan pe-
    if kata.startswith('pe'):
        kata_tanpa_awalan = kata[2:]
        if kata.startswith('peng'):
            kata_tanpa_awalan = kata[4:]
        elif kata.startswith('peny'):
            kata_tanpa_awalan = 's' + kata[4:]
        elif kata.startswith('pen'):
            kata_tanpa_awalan = kata[3:]
        elif kata.startswith('pem'):
            kata_tanpa_awalan = 'p' + kata[3:]
        if cek_kata_dasar(kata_tanpa_awalan, kamus):
            return kata_tanpa_awalan

    # Cek awalan te-
    if kata.startswith('te'):
        kata_tanpa_awalan = kata[2:]
        if kata.startswith('ter'):
            kata_tanpa_awalan = kata[3:]
        if cek_kata_dasar(kata_tanpa_awalan, kamus):
            return kata_tanpa_awalan

    return kata

def stem_kata(kata, kamus):
    # Implementasi algoritma Nazief-Adriani
    if cek_kata_dasar(kata, kamus):
        return kata

    # Hapus inflectional suffixes
    kata_1 = hapus_inflection_suffixes(kata)
    if cek_kata_dasar(kata_1, kamus):
        return kata_1

    # Hapus derivational suffix
    kata_2 = hapus_derivation_suffixes(kata_1)
    if cek_kata_dasar(kata_2, kamus):
        return kata_2

    # Hapus derivational prefix
    kata_3 = hapus_derivation_prefix(kata_2, kamus)
    if cek_kata_dasar(kata_3, kamus):
        return kata_3

    return kata


# Memuat kamus kata dasar
kamus = load_kamus()
def stem_text(text):

    # Memisahkan teks menjadi kata-kata
    kata_kata = text.split()

    # Melakukan stemming untuk setiap kata
    hasil_stemming = []
    for kata in kata_kata:
        kata_dasar = stem_kata(kata, kamus)
        hasil_stemming.append(kata_dasar)

    # Menggabungkan kembali kata-kata yang telah di-stemming
    return ' '.join(hasil_stemming)

# Remove emojis and special characters
def remove_emojis_and_symbols(text):
    if isinstance(text, str):
        # Remove emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        # Remove special characters and symbols, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    return text

stopwords = get_stopwords()
# Tokenization, stopword removal and stemming
def preprocess_text(text):
    # Tokenize
    tokens = text.split()

    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords]

    # Apply stemming
    tokens = [stem_text(word) for word in tokens]

    return ' '.join(tokens)

#LABELING
def load_lexicon(file_path):
    lexicon = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                word, score = line.strip().split('\t')
                lexicon[word] = int(score)
            except ValueError:
                continue
    return lexicon

positive_lexicon = load_lexicon('positive.tsv')
negative_lexicon = load_lexicon('negative.tsv')
all_lexicon = {**positive_lexicon, **negative_lexicon}

def classify_sentiment(score):
    if score > 0:
        return "Positif"
    elif score < 0:
        return "Negatif"
    else:
        return "Netral"

def calculate_polarity(words, lexicon):
    total_score = 0
    for word in words:
        total_score += lexicon.get(word, 0)
    return total_score

#VISUALISASI 
def process_all(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = remove_emojis_and_symbols(text)
    text = preprocess_text(text)
    return text

@app.route('/progress', methods=['GET'])
def get_progress():
    return jsonify(progress_data)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Ensure static directory exists on each request
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR)
    
    if request.method == 'POST':
        try:
            reset_progress()
            youtube_url = request.form.get('youtube_url')
            video_id = extract_video_id(youtube_url)
            
            # Step 1: Fetch Comments
            update_progress(1, "Fetching YouTube comments...")
            comments = video_comments(video_id)
            
            # Step 2: Process Text
            update_progress(2, "Processing text...")
            df = pd.DataFrame(comments, columns=['Comment'])
            
            # Apply the processing function using parallel processing
            # from multiprocessing import Pool, cpu_count
            # with Pool(processes=cpu_count()) as pool:
            #     df['processed_text'] = pool.map(process_all, df['Comment'])
            # Sequential processing instead of parallel
            df['processed_text'] = df['Comment'].apply(process_all)
            
            # Step 3: Sentiment Analysis
            update_progress(3, "Analyzing sentiment...")
            df['polarity_score'] = df['processed_text'].apply(lambda x: calculate_polarity(x.split(), all_lexicon))
            df['sentiment'] = df['polarity_score'].apply(classify_sentiment)
            
            # Step 4: Visualizations
            update_progress(4, "Generating visualizations...")
            # Hitung jumlah komentar berdasarkan sentimen
            sentiment_counts = df['sentiment'].value_counts()
            total_positive = sentiment_counts.get('Positif', 0)
            total_negative = sentiment_counts.get('Negatif', 0)
            total_neutral = sentiment_counts.get('Netral', 0)

            # Analisis frekuensi kata
            all_words = ' '.join(df['processed_text']).split()
            word_freq = Counter(all_words).most_common(10)  # Ambil hanya 10 kata populer
            labels, counts = zip(*word_freq)
            total_count = sum(counts)
            counts_percentage = [(count / total_count) * 100 for count in counts]

            # Sentiment Distribution (Pie Chart)
            mpl.rcParams['agg.path.chunksize'] = 10000
            plt.figure(figsize=(8, 8))
            plt.pie([total_positive, total_negative, total_neutral], 
                   labels=['Positive', 'Negative', 'Neutral'],
                   autopct='%1.1f%%',
                   colors=['#28a745', '#dc3545', '#007bff'],
                   startangle=90)
            plt.title('Sentiment Distribution')
            sentiment_path = os.path.join('static', 'sentiment_distribution.png')
            plt.savefig(sentiment_path, bbox_inches='tight', dpi=100)
            plt.close()

            # Word Cloud
            youtube_logo = np.array(Image.open(os.path.join('static', 'youtube_logo.png')))
            wordcloud = WordCloud(
                width=1000, 
                height=600, 
                background_color='white',
                mask=youtube_logo,
                contour_color='red',
                contour_width=2
            ).generate(' '.join(all_words))
            wordcloud_path = os.path.join('static', 'wordcloud.png')
            wordcloud.to_file(wordcloud_path)

            # Popular Words Distribution (Vertical Bar Chart)
            plt.figure(figsize=(10, 8))
            plt.barh(labels[::-1], counts[::-1])
            plt.xlabel('Frequency')
            plt.ylabel('Words')
            plt.title('Popular Words Distribution')
            plt.tight_layout()
            pie_chart_path = os.path.join('static', 'pie_chart.png')
            plt.savefig(pie_chart_path, bbox_inches='tight', dpi=100)
            plt.close()

            progress_data['status'] = 'complete'
            return render_template('index.html', 
                                total_positive=total_positive,
                                total_negative=total_negative,
                                total_neutral=total_neutral,
                                word_freq=word_freq,
                                sentiment_image=sentiment_path,
                                pie_chart_image=pie_chart_path,
                                wordcloud_image=wordcloud_path)
            
        except Exception as e:
            progress_data['status'] = 'error'
            progress_data['error'] = str(e)
            return render_template('index.html', error=f"Error: {e}", total_positive=None,
                         total_negative=None,
                         total_neutral=None,
                         word_freq=None,
                         sentiment_image=None,
                         pie_chart_image=None,
                         wordcloud_image=None)

    return render_template('index.html',
                         total_positive=None,
                         total_negative=None,
                         total_neutral=None,
                         word_freq=None,
                         sentiment_image=None,
                         pie_chart_image=None,
                         wordcloud_image=None)

# Configure temp directory for Render
# if os.environ.get('RENDER'):
#     tempfile.tempdir = '/tmp'

# Create static directory
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Update save_visualizations function
# def save_visualizations(sentiment_counts, word_freq, all_words):
#     try:
#         # Use temporary files for thread safety
#         with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_sentiment, \
#              tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_wordcloud, \
#              tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_barchart:
            
#             # Sentiment Distribution
#             plt.figure(figsize=(8, 8))
#             plt.pie(sentiment_counts, 
#                    labels=['Positive', 'Negative', 'Neutral'],
#                    autopct='%1.1f%%',
#                    colors=['#28a745', '#dc3545', '#007bff'])
#             plt.title('Sentiment Distribution')
#             plt.savefig(tmp_sentiment.name)
#             plt.close()

#             # Word Cloud
#             wordcloud = WordCloud(width=800, height=400, 
#                                 background_color='white').generate(' '.join(all_words))
#             wordcloud.to_file(tmp_wordcloud.name)

#             # Word Frequency Bar Chart
#             plt.figure(figsize=(10, 6))
#             labels, counts = zip(*word_freq)
#             plt.barh(labels[::-1], counts[::-1])
#             plt.title('Popular Words Distribution')
#             plt.xlabel('Frequency')
#             plt.tight_layout()
#             plt.savefig(tmp_barchart.name)
#             plt.close()

#             # Move files to static directory
#             os.rename(tmp_sentiment.name, os.path.join(STATIC_DIR, 'sentiment_distribution.png'))
#             os.rename(tmp_wordcloud.name, os.path.join(STATIC_DIR, 'wordcloud.png'))
#             os.rename(tmp_barchart.name, os.path.join(STATIC_DIR, 'pie_chart.png'))
            
#     except Exception as e:
#         print(f"Error in save_visualizations: {e}")
#         raise

# @app.route('/static/<path:filename>')
# def serve_static(filename):
#     return send_from_directory('static', filename)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
