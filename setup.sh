#!/bin/bash
echo "🚀 Setting up Bank Risk AI Agent environment..."

# Step 1. Create and activate virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "⚙️ Activating environment..."
source venv/bin/activate

# Step 2. Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Step 3. Install dependencies
if [ -f "requirements.txt" ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
else
    echo "❌ Error: requirements.txt not found!"
    exit 1
fi

# Step 4. Handle .env file
if [ ! -f ".env" ]; then
    echo "🧩 Creating .env configuration..."
    echo ""
    read -p "👉 Enter your IBM Watsonx API Key: " api_key
    read -p "👉 Enter your IBM Watsonx Project ID: " project_id
    read -p "👉 (Optional) Enter Watsonx URL [default: https://us-south.ml.cloud.ibm.com]: " watsonx_url

    # default URL fallback
    if [ -z "$watsonx_url" ]; then
        watsonx_url="https://us-south.ml.cloud.ibm.com"
    fi

    # write to .env
    echo "# IBM Watsonx Credentials" > .env
    echo "WATSONX_API_KEY=$api_key" >> .env
    echo "WATSONX_PROJECT_ID=$project_id" >> .env
    echo "WATSONX_URL=$watsonx_url" >> .env
    echo "" >> .env
    echo "✅ .env file created successfully."
else
    echo "✅ .env already exists. Skipping input."
fi

# Step 5. Launch Streamlit app
echo ""
echo "🚀 Launching Streamlit app..."
streamlit run app.py
