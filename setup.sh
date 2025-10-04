
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

# Step 4. Create .env file if not exists
if [ ! -f ".env" ]; then
    echo "🧩 Creating .env from template..."
    cp .env.example .env
else
    echo "✅ .env already exists."
fi

# Step 5. Launch Streamlit app
echo "🚀 Launching Streamlit app..."
streamlit run app.py
