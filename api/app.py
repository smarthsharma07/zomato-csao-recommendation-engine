import sys
import os

# Add required paths to sys.path so we can import modules
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "1_Model_Development"))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime

# Import inference engine
from online_api.inference import TwoStageEngine

# Initialize FastAPI app
app = FastAPI(title="Zomato CSAO Recommendation API")

# Load model engine eagerly at startup to ensure P99 < 200ms latency
print("Loading Recommendation Engine into memory...")
engine = TwoStageEngine()
print("Engine loaded successfully!")

class RecommendationRequest(BaseModel):
    cart_items: List[str]

# Minimal Frontend HTML
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zomato Recommendations</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; color: #333; margin: 0; padding: 0; display: flex; flex-direction: column; align-items: center; }
        .header { background-color: #e23744; color: white; width: 100%; padding: 20px 0; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { margin: 0; font-size: 24px; }
        .container { max-width: 800px; width: 90%; margin: 30px auto; background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
        .input-group { margin-bottom: 20px; }
        label { font-weight: bold; display: block; margin-bottom: 8px; color: #555; }
        input[type="text"] { width: 100%; padding: 12px; border: 1px solid #ccc; border-radius: 8px; font-size: 16px; box-sizing: border-box; }
        .helper-text { font-size: 13px; color: #777; margin-top: 5px; }
        button { background-color: #e23744; color: white; border: none; padding: 12px 24px; font-size: 16px; font-weight: bold; border-radius: 8px; cursor: pointer; transition: background 0.3s; width: 100%; }
        button:hover { background-color: #cb202d; }
        .loading { display: none; text-align: center; margin-top: 20px; font-weight: bold; color: #e23744; }
        .results { margin-top: 30px; }
        .result-card { display: flex; justify-content: space-between; align-items: center; padding: 15px; border: 1px solid #eee; border-radius: 8px; margin-bottom: 10px; background-color: #fafafa; transition: transform 0.2s; }
        .result-card:hover { transform: translateY(-2px); box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
        .item-name { font-weight: bold; font-size: 18px; }
        .item-score { background-color: #e23744; color: white; padding: 4px 10px; border-radius: 20px; font-size: 14px; font-weight: bold; }
        .context-info { margin-top: 20px; padding: 15px; background-color: #f0f8ff; border-radius: 8px; font-size: 14px; color: #444; }
        .error { color: red; text-align: center; margin-top: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🍽️ Zomato AI Cart Recommendations</h1>
    </div>
    
    <div class="container">
        <div class="input-group">
            <label for="cartInput">Current Cart Items (comma separated)</label>
            <input type="text" id="cartInput" placeholder="e.g. Butter Chicken, Garlic Naan" value="Butter Chicken, Garlic Naan">
            <div class="helper-text">Add multiple items separated by commas to simulate a user's cart.</div>
        </div>
        
        <button onclick="getRecommendations()">Get Recommendations ✨</button>
        
        <div class="loading" id="loading">Generating smart recommendations via Two-Stage ML... ⏳</div>
        <div class="error" id="error"></div>
        
        <div class="results" id="results"></div>
    </div>

    <script>
        async function getRecommendations() {
            const cartText = document.getElementById('cartInput').value;
            const items = cartText.split(',').map(item => item.trim()).filter(item => item.length > 0);
            
            if (items.length === 0) {
                alert("Please enter at least one item.");
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            document.getElementById('error').innerHTML = '';
            
            try {
                const startTime = performance.now();
                const response = await fetch('/api/recommend', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ cart_items: items })
                });
                
                const data = await response.json();
                const latency = Math.round(performance.now() - startTime);
                
                if (data.status === 'success') {
                    renderResults(data, latency);
                } else {
                    document.getElementById('error').innerText = "Error: " + data.message;
                }
            } catch (err) {
                document.getElementById('error').innerText = "Failed to reach the API. Is the server running?";
                console.error(err);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        function renderResults(data, latency) {
            const resultsDiv = document.getElementById('results');
            
            let html = `<h3>Top Recommendations (Latency: ${latency}ms)</h3>`;
            
            if (data.recommendations && data.recommendations.length > 0) {
                data.recommendations.forEach(rec => {
                    const score = (rec.score * 100).toFixed(1);
                    html += `
                        <div class="result-card">
                            <span class="item-name">${rec.item}</span>
                            <span class="item-score">${score}% Match</span>
                        </div>
                    `;
                });
            } else {
                html += `<p>No recommendations found for this combination.</p>`;
            }
            
            html += `
                <div class="context-info">
                    <strong>Inferred Context:</strong><br>
                    Time of Day: ${data.inferred_context.time_of_day} <br>
                    User Segment: ${data.inferred_context.user_segment}
                </div>
            `;
            
            resultsDiv.innerHTML = html;
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTMLResponse(content=HTML_TEMPLATE, status_code=200)

@app.post("/api/recommend")
async def recommend(request: RecommendationRequest):
    try:
        # Detect Time of Day automatically
        hour = datetime.now().hour
        if hour < 17: time_of_day = "Lunch"
        else: time_of_day = "Dinner"
        
        # Default user segment for the quick demo
        user_segment = "Premium" 
        
        # We pre-loaded the engine, so inference should just be standard forward passes
        # <200ms target should easily be met
        results = engine.recommend(
            request.cart_items, 
            user_segment, 
            time_of_day
        )
        return {
            "cart": request.cart_items,
            "inferred_context": {
                "time_of_day": time_of_day,
                "user_segment": user_segment
            },
            "recommendations": results,
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
