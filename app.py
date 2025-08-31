import os
import io
import csv
import json
import pandas as pd
from flask import Flask, render_template_string, request, jsonify, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
import base64
import speech_recognition as sr
import tempfile
from pydub import AudioSegment

# --- Pydub path configuration ---
try:
    AudioSegment.converter = r"E:\apps\ffmpeg-8.0\bin\ffmpeg.exe"
    AudioSegment.ffprobe = r"E:\apps\ffmpeg-8.0\bin\ffprobe.exe"
except Exception as e:
    print(f"Warning: Could not set FFmpeg paths. Voice analysis may fail. Error: {e}")

app = Flask(__name__)

# --- Step 1: Load and Prepare Training Data from CSV ---
try:
    train_data_path = os.path.join(os.path.dirname(__file__), 'train.csv')
    df = pd.read_csv(train_data_path)
    df.dropna(subset=['Review Text'], inplace=True)

    def map_rating_to_sentiment(rating):
        if rating in [4, 5]:
            return 'Positive'
        elif rating == 3:
            return 'Neutral'
        elif rating in [1, 2]:
            return 'Negative'
        return None

    df['sentiment'] = df['Rating'].apply(map_rating_to_sentiment)
    df.dropna(subset=['sentiment'], inplace=True)

    X = df['Review Text']
    y = df['sentiment']

    # --- Step 2: Train ML Model (Logistic Regression with TF-IDF) ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("logreg", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)

    # Calculate and store model accuracy for display
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)
    model_info = {"accuracy": f"{model_accuracy:.2f}"}

    print("Model trained successfully on train.csv!")
    print(f"Model accuracy on test set: {model_accuracy:.2f}")

except FileNotFoundError:
    print("Error: 'train.csv' not found. Please ensure your training data file is in the same directory.")
    X = ["sample review for training"]
    y = ["Neutral"]
    model = Pipeline([("tfidf", TfidfVectorizer()), ("logreg", LogisticRegression())])
    model.fit(X, y)
    model_info = {"accuracy": "N/A"}

except KeyError as e:
    print(f"Error: A required column was not found in the CSV file. Missing column: {e}")
    X = ["sample review for training"]
    y = ["Neutral"]
    model = Pipeline([("tfidf", TfidfVectorizer()), ("logreg", LogisticRegression())])
    model.fit(X, y)
    model_info = {"accuracy": "N/A"}

# --- Speech Recognition Setup ---
r = sr.Recognizer()

# --- Updated HTML Template with 'Shopping' and 'Voice' sections ---
template = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shopinion - AI Sentiment Analysis</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen flex flex-col">
    <nav class="bg-gray-800 shadow-md p-4 sticky top-0 z-50">
        <div class="container mx-auto flex justify-between items-center">
            <div class="text-2xl font-bold text-indigo-400">🛍️ Shopinion</div>
            <div class="space-x-4">
                <a href="#" id="analyze-link" class="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium">Analyze</a>
                <a href="#" id="shopping-link" class="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium">Shopping</a>
                <a href="#" id="voice-link" class="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium">Voice</a>
                <a href="#" id="about-btn" class="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium">About</a>
            </div>
        </div>
    </nav>

    <main class="flex-grow container mx-auto p-4 flex flex-col items-center justify-center">
        <div id="analyze-page" class="page active flex flex-col items-center justify-center text-center py-12 px-4 w-full max-w-4xl">
            <h2 class="text-3xl md:text-4xl font-bold text-gray-100 mb-8">Analyze Reviews</h2>
            <div class="w-full grid grid-cols-1 md:grid-cols-2 gap-8">
                <div class="bg-gray-800 p-8 rounded-2xl shadow-xl flex flex-col items-center col-span-1 md:col-span-2">
                    <h3 class="text-xl font-semibold mb-4">Live Review Analysis</h3>
                    <textarea id="live-review-input" class="w-full p-3 rounded-lg bg-gray-700 text-sm mb-4" rows="3" placeholder="Enter a single review here for instant sentiment prediction..."></textarea>
                    <div id="live-sentiment-result" class="text-lg font-bold"></div>
                </div>

                <div class="bg-gray-800 p-8 rounded-2xl shadow-xl flex flex-col items-center">
                    <h3 class="text-xl font-semibold mb-4">Manual Entry</h3>
                    <label for="review-count" class="text-sm mb-2">Enter number of reviews:</label>
                    <div class="flex items-center space-x-2">
                        <input type="number" id="review-count" min="1" class="w-20 p-2 text-center rounded-lg border border-gray-600 bg-gray-700">
                        <button id="generate-fields-btn" class="bg-indigo-600 px-4 py-2 rounded-lg">Generate</button>
                    </div>
                    <div id="manual-reviews-container" class="mt-6 w-full space-y-4"></div>
                </div>

                <div class="bg-gray-800 p-8 rounded-2xl shadow-xl flex flex-col items-center">
                    <h3 class="text-xl font-semibold mb-4">Upload CSV</h3>
                    <label for="csv-upload" class="cursor-pointer bg-gray-700 px-4 py-2 rounded-lg">Choose File</label>
                    <input type="file" id="csv-upload" accept=".csv" class="hidden">
                    <span id="file-name" class="mt-2 text-sm">No file chosen</span>
                    </div>
            </div>

            <button id="analyze-btn" class="mt-8 bg-green-600 px-8 py-3 rounded-full" disabled>Analyze</button>
            <div id="loading-spinner" class="mt-4 hidden animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-indigo-500"></div>
        </div>

        <div id="voice-page" class="page hidden flex flex-col items-center justify-center text-center py-12 px-4 w-full max-w-2xl">
            <h2 class="text-3xl md:text-4xl font-bold text-gray-100 mb-8">Voice to Text Sentiment Analysis</h2>
            <div class="bg-gray-800 p-8 rounded-2xl shadow-xl w-full flex flex-col items-center">
                <h3 class="text-xl font-semibold mb-4">Upload Audio File</h3>
                <p class="text-sm text-gray-400 mb-4">Supported format: WAV</p>
                <label for="audio-upload" class="cursor-pointer bg-indigo-600 px-6 py-3 rounded-lg text-lg">Choose Audio File</label>
                <input type="file" id="audio-upload" accept=".wav,.mp3" class="hidden">
                <span id="audio-file-name" class="mt-4 text-sm">No file chosen</span>
                <button id="transcribe-btn" class="mt-6 bg-green-600 px-8 py-3 rounded-full hidden">Transcribe & Analyze</button>
                <div id="voice-loading-spinner" class="mt-4 hidden animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-indigo-500"></div>
            </div>
            <div id="voice-results-container" class="hidden w-full bg-gray-800 p-8 rounded-2xl shadow-xl mt-8 text-left">
                <h3 class="text-xl font-semibold mb-4">Analysis Result</h3>
                <div class="bg-gray-700 p-4 rounded-lg mb-4">
                    <p class="text-sm font-semibold text-gray-400">Transcribed Text:</p>
                    <p id="transcribed-text" class="mt-2 text-gray-100 italic"></p>
                </div>
                <div class="flex items-center space-x-2">
                    <p class="text-lg font-semibold">Predicted Sentiment:</p>
                    <span id="voice-sentiment-result" class="text-xl font-bold"></span>
                </div>
            </div>
        </div>

        <div id="results-page" class="page hidden py-12 w-full max-w-5xl">
            <h2 class="text-3xl font-bold mb-8 text-center">Analysis Results</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div class="bg-gray-800 p-6 rounded-2xl shadow-xl flex flex-col items-center">
                    <h3 class="text-xl font-semibold mb-4">Overall Sentiment</h3>
                    <div id="sentiment-percentages" class="w-full text-left space-y-2 mb-4"></div>
                    <canvas id="sentiment-chart" class="w-full max-w-sm"></canvas>
                </div>
                <div class="bg-gray-800 p-6 rounded-2xl shadow-xl flex flex-col items-center">
                    <h3 class="text-xl font-semibold mb-4">Common Words</h3>
                    <img id="wordcloud-image" class="w-full h-auto mt-4">
                </div>
            </div>
            <div class="bg-gray-800 p-6 rounded-2xl shadow-xl mt-8">
                <h3 class="text-xl font-semibold mb-4">Individual Reviews</h3>
                <div id="individual-results" class="space-y-4 max-h-96 overflow-y-auto"></div>
            </div>

            <div class="text-center mt-8 space-x-4">
                <button id="back-to-start-btn" class="bg-gray-600 px-6 py-2 rounded-full">&larr; Analyze More</button>
                <button id="download-csv-btn" class="bg-indigo-600 px-6 py-2 rounded-full hidden">Download CSV</button>
            </div>
        </div>

        <div id="shopping-page" class="page hidden py-12 w-full max-w-4xl">
            <h2 class="text-3xl md:text-4xl font-bold text-gray-100 mb-8 text-center">Popular Shopping Websites</h2>
            <div id="shopping-cards-container" class="space-y-6">
                </div>
        </div>
    </main>

    <footer class="bg-gray-800 p-4 mt-auto text-center">
        <p class="text-gray-400 text-sm">© 2025 Shopinion | All Rights Reserved</p>
    </footer>

    <div id="about-modal" class="hidden fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-50">
        <div class="bg-gray-800 p-8 rounded-lg shadow-xl max-w-lg w-full text-center relative">
            <button id="close-modal-btn" class="absolute top-4 right-4 text-gray-400 hover:text-white">&times;</button>
            <h3 class="text-2xl font-bold mb-4">About Shopinion</h3>
            <p class="text-gray-300 leading-relaxed mb-4">
                Shopinion is a powerful sentiment analysis tool designed to help you understand customer feedback instantly.
                Using a machine learning model trained on real review data, it can accurately classify a review as
                <span class="text-green-400 font-semibold">Positive</span>,
                <span class="text-red-400 font-semibold">Negative</span>, or
                <span class="text-gray-400 font-semibold">Neutral</span>.
                Simply enter your reviews manually or upload a CSV file, and get a detailed breakdown of the overall sentiment.
            </p>
            <h3 class="text-xl font-semibold mb-2">Model Information</h3>
            <p class="text-gray-300">Accuracy on test data: <span id="model-accuracy" class="font-bold text-green-400"></span></p>
        </div>
    </div>

    <script>
        const pageElements = {
            analyze: document.getElementById('analyze-page'),
            results: document.getElementById('results-page'),
            shopping: document.getElementById('shopping-page'),
            voice: document.getElementById('voice-page')
        };
        const analyzeLink = document.getElementById('analyze-link');
        const shoppingLink = document.getElementById('shopping-link');
        const voiceLink = document.getElementById('voice-link');
        const analyzeBtn = document.getElementById('analyze-btn');
        const backToStartBtn = document.getElementById('back-to-start-btn');
        const downloadCsvBtn = document.getElementById('download-csv-btn');
        const reviewCountInput = document.getElementById('review-count');
        const generateFieldsBtn = document.getElementById('generate-fields-btn');
        const manualReviewsContainer = document.getElementById('manual-reviews-container');
        const csvUploadInput = document.getElementById('csv-upload');
        const fileNameSpan = document.getElementById('file-name');
        const csvColumnSelectContainer = document.getElementById('csv-column-select-container');
        const csvColumnSelect = document.getElementById('csv-column-select');
        const loadingSpinner = document.getElementById('loading-spinner');
        const sentimentPercentagesDiv = document.getElementById('sentiment-percentages');
        const aboutBtn = document.getElementById('about-btn');
        const aboutModal = document.getElementById('about-modal');
        const closeModalBtn = document.getElementById('close-modal-btn');
        const liveReviewInput = document.getElementById('live-review-input');
        const liveSentimentResult = document.getElementById('live-sentiment-result');
        const shoppingCardsContainer = document.getElementById('shopping-cards-container');
        const audioUploadInput = document.getElementById('audio-upload');
        const audioFileNameSpan = document.getElementById('audio-file-name');
        const transcribeBtn = document.getElementById('transcribe-btn');
        const voiceLoadingSpinner = document.getElementById('voice-loading-spinner');
        const voiceResultsContainer = document.getElementById('voice-results-container');
        const transcribedTextDiv = document.getElementById('transcribed-text');
        const voiceSentimentResultDiv = document.getElementById('voice-sentiment-result');

        let currentInputMethod = null;
        let csvFile = null;
        let analysisData = [];
        let myChart = null;

        const shoppingSites = [
            {
                name: "Amazon",
                url: "https://www.amazon.com",
                description: "The world's largest online retailer, offering a vast selection of products from books to electronics.",
                image: "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg"
            },
            {
                name: "eBay",
                url: "https://www.ebay.com",
                description: "An e-commerce giant known for its auctions and 'Buy It Now' sales of new and used goods.",
                image: "https://upload.wikimedia.org/wikipedia/commons/4/48/EBay_logo.png"
            },
            {
                name: "Walmart",
                url: "https://www.walmart.com",
                description: "A multinational retail corporation operating a chain of hypermarkets, discount department stores, and grocery stores.",
                image: "https://static.vecteezy.com/system/resources/previews/018/930/234/non_2x/walmart-transparent-logo-free-png.png"
            },
            {
                name: "Target",
                url: "https://www.target.com",
                description: "A major American retail corporation that sells a wide range of products, including clothing, home goods, and electronics.",
                image: "https://download.logo.wine/logo/Target_Corporation/Target_Corporation-Logo.wine.png"
            },
            {
                name: "Flipkart",
                url: "https://www.flipkart.com",
                description: "India's leading e-commerce company, offering a wide range of products from electronics to fashion.",
                image: "https://tse3.mm.bing.net/th/id/OIP.OynH-tdXa4WwFNN6pvylVQHaHa?rs=1&pid=ImgDetMain&o=7&rm=3"
            },
            {
                name: "Myntra",
                url: "https://www.myntra.com",
                description: "A major Indian fashion e-commerce company, focusing on clothing, footwear, and accessories.",
                image: "https://cdn.iconscout.com/icon/free/png-512/myntra-2709168-2249158.png"
            },
            {
                name: "Meesho",
                url: "https://www.meesho.com",
                description: "An Indian social commerce platform that enables small businesses and individuals to start their online stores via social channels.",
                image: "https://cdn.freelogovectors.net/wp-content/uploads/2023/11/meesho-logo-01_freelogovectors.net_.png"
            },
            {
                name: "Snapdeal",
                url: "https://www.snapdeal.com",
                description: "An Indian e-commerce company that sells a diverse range of products from various categories.",
                image: "https://tse3.mm.bing.net/th/id/OIP.e8-DUCxXwWxQQivtxj39PgAAAA?rs=1&pid=ImgDetMain&o=7&rm=3"
            }
        ];

        const showPage = (pageName) => {
            Object.values(pageElements).forEach(p => p.classList.add('hidden'));
            pageElements[pageName].classList.remove('hidden');
        };

        const renderShoppingCards = () => {
            shoppingCardsContainer.innerHTML = shoppingSites.map(site => `
                <div class="bg-gray-800 p-6 rounded-2xl shadow-xl flex items-center space-x-6">
                    <div class="flex-shrink-0 w-16 h-16 bg-white rounded-lg flex items-center justify-center p-2">
                        <img src="${site.image}" alt="${site.name} logo" class="max-w-full max-h-full object-contain">
                    </div>
                    <div class="flex-1">
                        <a href="${site.url}" target="_blank" class="text-xl font-semibold text-indigo-400 hover:underline">${site.name}</a>
                        <p class="text-sm text-gray-400 mt-1">${site.description}</p>
                    </div>
                </div>
            `).join('');
        };
        
        // Initial page load
        document.addEventListener('DOMContentLoaded', () => {
            showPage('analyze');
            renderShoppingCards();
        });

        // Event listeners for navigation links
        analyzeLink.addEventListener('click', (e) => {
            e.preventDefault();
            showPage('analyze');
        });
        
        shoppingLink.addEventListener('click', (e) => {
            e.preventDefault();
            showPage('shopping');
        });

        voiceLink.addEventListener('click', (e) => {
            e.preventDefault();
            showPage('voice');
            // Reset voice page elements
            audioUploadInput.value = '';
            audioFileNameSpan.textContent = 'No file chosen';
            transcribeBtn.classList.add('hidden');
            voiceResultsContainer.classList.add('hidden');
        });
        
        // Event listeners for About modal
        aboutBtn.addEventListener('click', (e) => {
            e.preventDefault();
            aboutModal.classList.remove('hidden');
            fetchModelAccuracy();
        });

        closeModalBtn.addEventListener('click', () => {
            aboutModal.classList.add('hidden');
        });

        aboutModal.addEventListener('click', (e) => {
            if (e.target === aboutModal) {
                aboutModal.classList.add('hidden');
            }
        });

        backToStartBtn.addEventListener('click', () => {
            showPage('analyze');
            analysisData = [];
            downloadCsvBtn.classList.add('hidden');
            fileNameSpan.textContent = 'No file chosen';
            manualReviewsContainer.innerHTML = '';
            reviewCountInput.value = '';
            csvUploadInput.value = '';
            // The column select container is now permanently hidden
            currentInputMethod = null;
            analyzeBtn.disabled = true;
        });

        generateFieldsBtn.addEventListener('click', () => {
            const count = parseInt(reviewCountInput.value, 10);
            if (count > 0) {
                manualReviewsContainer.innerHTML = '';
                for (let i = 0; i < count; i++) {
                    const reviewGroup = document.createElement('div');
                    reviewGroup.classList.add('space-y-2', 'p-4', 'border', 'border-gray-700', 'rounded-lg', 'bg-gray-800');
                    const textarea = document.createElement('textarea');
                    textarea.placeholder = `Enter review #${i + 1}`;
                    textarea.classList.add('review-text', 'w-full','p-3','rounded-lg','bg-gray-700','text-sm');
                    textarea.rows = 3;
                    reviewGroup.appendChild(textarea);
                    manualReviewsContainer.appendChild(reviewGroup);
                }
                currentInputMethod = 'manual';
                analyzeBtn.disabled = false;
                csvUploadInput.value = '';
                fileNameSpan.textContent = 'No file chosen';
            }
        });

        csvUploadInput.addEventListener('change', async (e) => {
            csvFile = e.target.files[0];
            if (csvFile) {
                fileNameSpan.textContent = csvFile.name;
                manualReviewsContainer.innerHTML = '';
                reviewCountInput.value = '';
                analyzeBtn.disabled = false; // Enable analyze button as soon as a file is chosen
                currentInputMethod = 'csv';
            } else {
                fileNameSpan.textContent = 'No file chosen';
                analyzeBtn.disabled = true;
            }
        });

        analyzeBtn.addEventListener('click', async () => {
            if (!currentInputMethod) return;
            analyzeBtn.disabled = true;
            loadingSpinner.classList.remove('hidden');

            try {
                let response;
                if (currentInputMethod === 'manual') {
                    const textareas = manualReviewsContainer.querySelectorAll('.review-text');
                    const reviews = Array.from(textareas).map(textarea => textarea.value).filter(t => t.trim() !== '');
                    if (reviews.length === 0) {
                        throw new Error("No reviews to analyze. Please enter some reviews.");
                    }
                    response = await fetch('/analyze_reviews', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ reviews: reviews })
                    });
                } else if (currentInputMethod === 'csv' && csvFile) {
                    const formData = new FormData();
                    formData.append('csv_file', csvFile);
                    // The column name is hardcoded to "Review Text"
                    formData.append('column_name', 'Review Text');
                    response = await fetch('/analyze_reviews', {
                        method: 'POST',
                        body: formData
                    });
                } else {
                    throw new Error("Invalid input method.");
                }

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                if (result.error) {
                    throw new Error(result.error);
                }

                analysisData = result.analysis;
                renderResults(analysisData, result.wordcloud_img);
                showPage('results');
            } catch (e) {
                console.error("Analysis failed:", e);
                alert(e.message || "An error occurred during analysis. Please try again.");
            } finally {
                analyzeBtn.disabled = false;
                loadingSpinner.classList.add('hidden');
            }
        });

        downloadCsvBtn.addEventListener('click', async () => {
            if (analysisData.length === 0) return;
            const response = await fetch('/download_results', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ analysis: analysisData })
            });
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'sentiment_analysis_results.csv';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        });

        // Live review analysis
        liveReviewInput.addEventListener('input', async () => {
            const reviewText = liveReviewInput.value;
            if (reviewText.trim().length > 3) {
                try {
                    const response = await fetch('/predict_sentiment', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ review: reviewText })
                    });
                    const result = await response.json();
                    if (result.error) {
                        liveSentimentResult.textContent = 'Error';
                        liveSentimentResult.className = 'text-red-500 text-lg font-bold';
                    } else {
                        const sentiment = result.sentiment;
                        liveSentimentResult.textContent = `Predicted: ${sentiment}`;
                        if (sentiment === 'Positive') {
                            liveSentimentResult.className = 'text-green-400 text-lg font-bold';
                        } else if (sentiment === 'Negative') {
                            liveSentimentResult.className = 'text-red-400 text-lg font-bold';
                        } else {
                            liveSentimentResult.className = 'text-gray-400 text-lg font-bold';
                        }
                    }
                } catch (e) {
                    liveSentimentResult.textContent = 'Error predicting';
                    liveSentimentResult.className = 'text-red-500 text-lg font-bold';
                }
            } else {
                liveSentimentResult.textContent = '';
            }
        });

        async function fetchModelAccuracy() {
            try {
                const response = await fetch('/model_accuracy');
                const result = await response.json();
                document.getElementById('model-accuracy').textContent = result.accuracy;
            } catch (e) {
                document.getElementById('model-accuracy').textContent = 'N/A';
            }
        }

        function renderResults(analysis, wordcloudImg) {
            const individualResultsContainer = document.getElementById('individual-results');
            individualResultsContainer.innerHTML = '';
            const sentimentCounts = { Positive: 0, Negative: 0, Neutral: 0 };

            analysis.forEach(item => {
                if (item.sentiment in sentimentCounts) {
                    sentimentCounts[item.sentiment]++;
                }
            });

            analysis.forEach(item => {
                const div = document.createElement('div');
                const icon = item.sentiment === 'Positive' ? '😊' : item.sentiment === 'Negative' ? '😡' : '😐';
                div.classList.add('p-4','rounded-lg','bg-gray-700','flex','items-start','space-x-3');
                div.innerHTML = `<div class="text-xl mt-1">${icon}</div><div><p class="font-medium">${item.sentiment}</p><p class="text-sm italic mt-1">"${item.review}"</p></div>`;
                individualResultsContainer.appendChild(div);
            });

            const totalReviews = analysis.length;
            const percentages = {};
            for (const sentiment in sentimentCounts) {
                percentages[sentiment] = totalReviews > 0 ? (sentimentCounts[sentiment] / totalReviews) * 100 : 0;
            }

            sentimentPercentagesDiv.innerHTML = `
                <p class="text-sm"><span class="font-bold text-green-400">Positive:</span> ${percentages.Positive.toFixed(1)}%</p>
                <p class="text-sm"><span class="font-bold text-red-400">Negative:</span> ${percentages.Negative.toFixed(1)}%</p>
                <p class="text-sm"><span class="font-bold text-gray-400">Neutral:</span> ${percentages.Neutral.toFixed(1)}%</p>
            `;

            drawPieChart(sentimentCounts);
            document.getElementById('wordcloud-image').src = wordcloudImg;
            downloadCsvBtn.classList.remove('hidden');
        }

        function drawPieChart(counts) {
            const ctx = document.getElementById('sentiment-chart').getContext('2d');
            if (myChart) myChart.destroy();
            myChart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: ['Positive','Negative','Neutral'],
                    datasets: [{
                        data: [counts.Positive, counts.Negative, counts.Neutral],
                        backgroundColor:['rgb(74,222,128)','rgb(239,68,68)','rgb(156,163,175)']
                    }]
                }
            });
        }

        // Voice to Text functionality
        audioUploadInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                audioFileNameSpan.textContent = file.name;
                transcribeBtn.classList.remove('hidden');
                voiceResultsContainer.classList.add('hidden');
            } else {
                audioFileNameSpan.textContent = 'No file chosen';
                transcribeBtn.classList.add('hidden');
            }
        });

        transcribeBtn.addEventListener('click', async () => {
            const file = audioUploadInput.files[0];
            if (!file) return;

            transcribeBtn.disabled = true;
            voiceLoadingSpinner.classList.remove('hidden');
            voiceResultsContainer.classList.add('hidden');

            const formData = new FormData();
            formData.append('audio_file', file);

            try {
                const response = await fetch('/analyze_voice', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const result = await response.json();

                if (result.error) {
                    throw new Error(result.error);
                }

                transcribedTextDiv.textContent = result.transcribed_text;
                voiceSentimentResultDiv.textContent = result.sentiment;

                if (result.sentiment === 'Positive') {
                    voiceSentimentResultDiv.className = 'text-green-400 text-xl font-bold';
                } else if (result.sentiment === 'Negative') {
                    voiceSentimentResultDiv.className = 'text-red-400 text-xl font-bold';
                } else {
                    voiceSentimentResultDiv.className = 'text-gray-400 text-xl font-bold';
                }

                voiceResultsContainer.classList.remove('hidden');

            } catch (e) {
                alert('Error: ' + (e.message || 'An unknown error occurred.'));
            } finally {
                transcribeBtn.disabled = false;
                voiceLoadingSpinner.classList.add('hidden');
            }
        });

    </script>
</body>
</html>
"""

# --- Flask Routes ---
@app.route("/")
def home():
    return render_template_string(template)

@app.route("/model_accuracy")
def model_accuracy():
    return jsonify(model_info)

@app.route("/get_csv_headers", methods=["POST"])
def get_csv_headers():
    # This route is no longer needed since the dropdown is removed, but it's kept for completeness.
    if 'csv_file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['csv_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        df = pd.read_csv(file)
        headers = df.columns.tolist()
        return jsonify({"headers": headers})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/analyze_reviews", methods=["POST"])
def analyze_reviews():
    reviews = []
    wordcloud_img = None
    
    if 'csv_file' in request.files:
        file = request.files['csv_file']
        # The column name is now hardcoded to "Review Text"
        column_name = "Review Text"
        try:
            df = pd.read_csv(file)
            if column_name and column_name in df.columns:
                df_reviews = df[column_name].dropna().astype(str)
                reviews = df_reviews.tolist()
            else:
                return jsonify({"error": "The required column 'Review Text' was not found in the CSV file."}), 400
        except Exception as e:
            return jsonify({"error": f"Error reading CSV: {str(e)}"}), 400
    else:
        data = request.get_json()
        if data:
            reviews = data.get("reviews", [])

    reviews = [r for r in reviews if r and r.strip()]
    if not reviews:
        return jsonify({"error": "No valid reviews to analyze."}), 400
    
    # Generate Word Cloud
    all_reviews_text = " ".join(reviews)
    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='viridis').generate(all_reviews_text)
    img_stream = io.BytesIO()
    wordcloud.to_image().save(img_stream, format='PNG')
    img_stream.seek(0)
    wordcloud_img = f"data:image/png;base64,{base64.b64encode(img_stream.read()).decode('utf-8')}"

    sentiments = model.predict(reviews)
    analysis_results = [{"review": review, "sentiment": sentiment} for review, sentiment in zip(reviews, sentiments)]

    return jsonify({"analysis": analysis_results, "wordcloud_img": wordcloud_img})

@app.route("/predict_sentiment", methods=["POST"])
def predict_sentiment():
    data = request.get_json()
    review = data.get("review", "")
    if not review or not review.strip():
        return jsonify({"error": "No review provided."}), 400
    
    sentiment = model.predict([review])[0]
    return jsonify({"sentiment": sentiment})

@app.route("/analyze_voice", methods=["POST"])
def analyze_voice():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file."}), 400
    
    try:
        # Create a temporary WAV file for SpeechRecognition
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            audio_path = temp_wav.name
            # Convert any format to WAV using pydub
            if audio_file.filename.lower().endswith('.mp3'):
                audio = AudioSegment.from_mp3(audio_file)
            elif audio_file.filename.lower().endswith('.wav'):
                audio = AudioSegment.from_wav(audio_file)
            else:
                return jsonify({"error": "Unsupported audio format. Please upload a WAV or MP3 file."}), 400
            
            audio.export(audio_path, format="wav")

        # Transcribe the WAV file
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
        
        transcribed_text = r.recognize_google(audio_data)

        if not transcribed_text:
            return jsonify({"error": "Could not transcribe the audio. The file might be empty or in a format not supported by the model."}), 400

        # Predict sentiment of the transcribed text
        sentiment = model.predict([transcribed_text])[0]
        
        return jsonify({
            "transcribed_text": transcribed_text,
            "sentiment": sentiment
        })
    except sr.UnknownValueError:
        return jsonify({"error": "Google Speech Recognition could not understand the audio. Please try a clearer audio file."}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Could not request results from Google Speech Recognition service; {e}"}), 500
    except FileNotFoundError:
        return jsonify({"error": "FFmpeg or avconv not found. Please ensure it's installed and in your system's PATH, or check the manual path in the Python code."}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    finally:
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)

@app.route("/download_results", methods=["POST"])
def download_results():
    data = request.get_json()
    analysis = data.get("analysis", [])

    if not analysis:
        return jsonify({"error": "No data to download"}), 400

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Review", "Sentiment"])
    for row in analysis:
        review_text = row["review"].replace('\\n', ' ').replace('\\r', ' ').strip()
        writer.writerow([review_text, row["sentiment"]])

    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype="text/csv",
        as_attachment=True,
        download_name="sentiment_analysis_results.csv"
    )

if __name__ == "__main__":
    app.run(debug=True)