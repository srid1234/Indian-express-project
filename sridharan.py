import streamlit as st

# Basic page config
st.set_page_config(
    page_title="Simple Test App",
    page_icon=":memo:"
)

# Main content
st.title("Simple Test App")
st.write("This is a test to ensure Streamlit is working properly.")

# Add a simple input
user_input = st.text_input("Enter some text:")

# Add a button
if st.button("Click me!"):
    if user_input:
        st.success(f"You entered: {user_input}")
    else:
        st.warning("Please enter some text first!")

def load_or_create_model():
    """Load existing model or create a dummy one for demonstration"""
    try:
        lr_model = joblib.load('logistic_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
    except FileNotFoundError:
        # Create dummy model for demonstration
        st.warning("⚠️ Using a demonstration model. For production, please train and save the actual model files.")
        
        # Create simple dummy data
        dummy_texts = [
            "Technology news about the latest gadgets",
            "Sports update from the recent match",
            "Business report on stock market"
        ]
        dummy_labels = ['Technology', 'Sports', 'Business']
        
        # Create and fit vectorizer
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(dummy_texts)
        
        # Create and fit label encoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(dummy_labels)
        
        # Create and fit model
        lr_model = LogisticRegression()
        lr_model.fit(X, y)
        
        # Save the models
        joblib.dump(lr_model, 'logistic_model.joblib')
        joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
        joblib.dump(label_encoder, 'label_encoder.joblib')
    
    return lr_model, vectorizer, label_encoder

# Load or create the model and components
lr_model, vectorizer, label_encoder = load_or_create_model()

# Create the user interface
st.title("📰 News Article Category Prediction")
st.write("Enter a news article to predict its category:")

# Add sample text button
if st.button("Load Sample Text"):
    sample_text = """Apple announces new iPhone with revolutionary AI capabilities. 
    The tech giant revealed its latest smartphone featuring advanced artificial intelligence 
    that can predict user behavior and optimize performance."""
    st.session_state.user_input = sample_text
else:
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

# User input text box
user_input = st.text_area("Input News Article:", value=st.session_state.user_input, height=150)

# Predict button with error handling
if st.button("🔍 Predict Category"):
    try:
        if user_input.strip():  # Check if the input is not empty
            # Show spinner while processing
            with st.spinner("Analyzing article..."):
                # Transform the input text using the vectorizer
                user_input_transformed = vectorizer.transform([user_input])
                
                # Get the model prediction
                prediction = lr_model.predict(user_input_transformed)
                
                # Decode the prediction
                predicted_category = label_encoder.inverse_transform(prediction)
                
                # Show the predicted category with styling
                st.success("✅ Prediction Complete!")
                st.subheader(f"📑 Predicted Category: {predicted_category[0]}")
        else:
            st.error("⚠️ Please enter a news article.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please try again with a different input or contact support if the issue persists.")
