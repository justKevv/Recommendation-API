# Student Recommendation API

A FastAPI-based machine learning system that predicts job categories and recommends internships based on student profiles using advanced NLP techniques and geolocation analysis.

## 🌟 Features

- **Job Category Prediction**: Uses TF-IDF vectorization and Random Forest classification to predict suitable job categories from student profile text
- **Intelligent Internship Recommendations**: Employs sentence transformers for semantic similarity matching between student profiles and internship descriptions
- **Location-Based Ranking**: Incorporates geographical proximity using geolocation data to enhance recommendations
- **Caching System**: Implements efficient geocoding cache to minimize API calls and improve performance
- **RESTful API**: Clean and well-documented FastAPI endpoints with automatic OpenAPI documentation

## 🏗️ Architecture

```
Recommendation-System/
├── app/
│   ├── api/
│   │   └── routes/
│   │       └── recommendations.py    # API endpoints
│   ├── core/
│   │   └── models.py                 # ML model loading and initialization
│   ├── schemas/
│   │   └── recommendation.py         # Pydantic data models
│   ├── services/
│   │   └── ranking.py               # Core recommendation logic
│   ├── utils/
│   │   └── text.py                  # Text preprocessing utilities
│   └── main.py                      # FastAPI application setup
├── models/                          # Pre-trained ML models
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   └── random_forest_model.pkl
├── GEO_CACHE.txt                   # Geocoding cache file
└── requirements.txt                # Python dependencies
```

## 🤖 Machine Learning Components

### 1. Job Category Prediction
- **TF-IDF Vectorizer**: Converts text to numerical features
- **Random Forest Classifier**: Predicts job categories from profile text
- **Label Encoder**: Maps category predictions back to human-readable labels

### 2. Internship Recommendation
- **Sentence Transformers**: Uses `all-MiniLM-L6-v2` model for semantic embeddings
- **Cosine Similarity**: Measures semantic similarity between profiles and internships
- **Two-Stage Ranking**: Combines semantic similarity with geographical proximity

### 3. Geolocation Enhancement
- **Geocoding**: Converts city names to coordinates using Nominatim API
- **Distance Calculation**: Uses Haversine formula via GeoPy for accurate distance measurement
- **Location Bonus**: Applies score boosts for nearby opportunities (same location: +1.0, <150km: +0.5)

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Pre-trained ML models (`.pkl` files) in the `models/` directory

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Recommendation-System
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are in place**:
   ```
   models/
   ├── tfidf_vectorizer.pkl
   ├── label_encoder.pkl
   └── random_forest_model.pkl
   ```

### Running the Application

1. **Start the development server**:
   ```bash
   uvicorn app.main:app --reload
   ```

2. **Access the API**:
   - API: http://localhost:8000
   - Interactive Documentation: http://localhost:8000/docs
   - Alternative Documentation: http://localhost:8000/redoc

## 📖 API Endpoints

### 1. Predict Job Category
**POST** `/api/v1/predict-category`

Predicts the most suitable job category based on a student's profile text.

**Request Body**:
```json
{
  "profile_text": "Computer science student with experience in Python programming, data analysis, and machine learning projects."
}
```

**Response**:
```json
{
  "predicted_category": "Data Science"
}
```

### 2. Recommend Internships
**POST** `/api/v1/recommend-internships`

Returns a ranked list of internship IDs based on profile matching and location preferences.

**Request Body**:
```json
{
  "profile_text": "Computer science student interested in web development",
  "predicted_category": "Software Development",
  "preferred_location": "Jakarta",
  "internships": [
    {
      "id": 1,
      "internship_text": "Frontend developer internship using React and JavaScript",
      "location": "Jakarta"
    },
    {
      "id": 2,
      "internship_text": "Data analyst internship with Python and SQL",
      "location": "Surabaya"
    }
  ]
}
```

**Response**:
```json
{
  "recommendations": [1, 2]
}
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory for configuration:
```env
# Optional: Configure API settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
```

### Geocoding Cache
The system automatically caches geocoding results in `GEO_CACHE.txt` to improve performance and reduce API calls to the Nominatim service.

## 🛠️ Development

### Project Structure Details

- **`app/main.py`**: FastAPI application initialization and configuration
- **`app/api/routes/`**: API endpoint definitions and routing
- **`app/core/models.py`**: ML model loading and global model instances
- **`app/services/ranking.py`**: Core business logic for recommendations
- **`app/schemas/`**: Pydantic models for request/response validation
- **`app/utils/`**: Utility functions for text processing

### Adding New Features

1. **New API Endpoints**: Add routes in `app/api/routes/`
2. **Data Models**: Define Pydantic schemas in `app/schemas/`
3. **Business Logic**: Implement services in `app/services/`
4. **Utilities**: Add helper functions in `app/utils/`

## 📦 Dependencies

- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for FastAPI
- **scikit-learn**: Machine learning library for classification
- **sentence-transformers**: State-of-the-art sentence embeddings
- **GeoPy**: Geocoding and distance calculations
- **Pydantic**: Data validation using Python type annotations
- **PyTorch**: Deep learning framework (dependency of sentence-transformers)

## 🌍 Geolocation Features

The system is specifically configured for Indonesian locations:
- Geocoding queries are appended with ", Indonesia" for better accuracy
- Distance calculations use the Haversine formula for precise geographical measurements
- Cached locations include major Indonesian cities for quick lookup

## 🔒 Production Considerations

1. **Model Files**: Ensure all `.pkl` files are properly versioned and backed up
2. **Error Handling**: The system gracefully handles missing models with informative error messages
3. **Caching**: Geocoding cache reduces external API dependencies
4. **Scalability**: Consider using Redis for geocoding cache in production environments
5. **Security**: Implement authentication and rate limiting for production deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions or issues, please:
1. Check the existing issues on GitHub
2. Create a new issue with detailed information
3. Include relevant error messages and system information

---

**Note**: This system requires pre-trained machine learning models. Ensure you have the necessary `.pkl` files in the `models/` directory before running the application.
