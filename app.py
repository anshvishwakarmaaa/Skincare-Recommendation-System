import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import sys
import json
import logging
import requests
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
PORT = int(os.environ.get('PORT', 5000))

# Settings from config
from config import CONDITIONS, MODEL_PATH, THRESHOLDS_PATH, UPLOADS_DIR, LOGS_DIR, API_BASE_URL, MAX_FILE_SIZE, ALLOWED_EXTENSIONS

# Setup logging
os.makedirs(LOGS_DIR, exist_ok=True)
logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(os.path.join(LOGS_DIR, "app.log"), maxBytes=5000000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from FixedSkinModel import FixedSkinModel
    import torchvision.transforms as transforms
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    raise

app = Flask(__name__, static_folder='frontend')
CORS(app, origins="*")

app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Serve frontend
@app.route('/')
def serve_frontend():
    return send_from_directory('frontend', 'index.html')

@app.route('/frontend/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

class SkinAnalyzerAPI:
    def __init__(self, model_path=MODEL_PATH):
        self.conditions = CONDITIONS
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = self.load_model(model_path)
        
        # Exact snippet requested:
        import json
        thresholds_path = 'models/optimal_thresholds.json'
        if os.path.exists(thresholds_path):
            with open(thresholds_path) as f:
                self.thresholds = json.load(f)
            print(f"✅ Loaded optimal thresholds: {self.thresholds}")
        else:
            self.thresholds = {cond: 0.5 for cond in self.conditions}
            print("⚠️ Using default 0.5 thresholds")
            
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        model = FixedSkinModel(num_classes=len(self.conditions))
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_image(self, image_pil):
        try:
            image_tensor = self.transform(image_pil).unsqueeze(0).float()
            return image_tensor.to(self.device)
        except Exception as e:
            logger.error(f"❌ Error preprocessing image: {e}")
            raise
    
    def analyze_skin(self, image_pil):
        try:
            with torch.no_grad():
                processed_image = self.preprocess_image(image_pil)
                outputs = self.model(processed_image)
                # Exact snippet requested:
                probabilities = F.sigmoid(outputs).cpu().numpy()[0]
            
            results = {}
            for condition, prob in zip(self.conditions, probabilities):
                # We can store the raw probabilities but UI uses thresholds to flag severity
                results[condition] = float(prob)
                
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during analysis: {e}")
            raise
            
    def generate_ai_recommendations(self, analysis_results, report):
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        # Use exact thresholds variable instead of hardcoded 0.3
        detected = {k: round(v*100, 1) for k, v in analysis_results.items() if v > self.thresholds.get(k, 0.5)}
        skin_type = report.get("skin_type", "Unknown")
        score = report.get("overall_score", 0)
        concerns = [c["condition"] for c in report.get("primary_concerns", [])]
        
        prompt = f"""You are a professional dermatologist and certified skincare expert.
A patient has completed an AI skin analysis. Based on the results below, provide highly personalized skincare recommendations.

SKIN ANALYSIS RESULTS:
- Overall Skin Health Score: {score}/100
- Detected Skin Type: {skin_type}
- Primary Concerns: {', '.join(concerns) if concerns else 'None'}
- Condition Confidence Scores: {detected}

Provide recommendations in the following strict JSON format only, no extra text, no markdown fences:
{{
  "morning_routine": [
    {{"step": 1, "product_type": "Gentle Cleanser", "reason": "...", "ingredient_to_look_for": "...", "ingredient_to_avoid": "..."}}
  ],
  "evening_routine": [
    {{"step": 1, "product_type": "...", "reason": "...", "ingredient_to_look_for": "...", "ingredient_to_avoid": "..."}}
  ],
  "weekly_treatments": [
    {{"treatment": "...", "frequency": "...", "benefit": "..."}}
  ],
  "lifestyle_tips": [
    {{"tip": "...", "impact": "..."}}
  ],
  "ingredients_to_prioritize": ["..."],
  "ingredients_to_avoid": ["..."],
  "urgency_note": "...",
  "follow_up_recommendation": "..."
}}

Rules:
- Morning routine must have 4-5 steps
- Evening routine must have 4-6 steps
- Weekly treatments must have 2-3 items
- Lifestyle tips must have 3-4 items
- Be specific to the detected conditions and confidence scores
- Mention specific active ingredients by name (e.g. niacinamide, retinol, salicylic acid)
- If score < 50 set urgency_note to strongly recommend seeing a dermatologist
- Keep all text concise and actionable
- Return pure JSON only, absolutely no extra text or markdown"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional dermatologist. You always respond with pure valid JSON only. No markdown, no explanation, no extra text whatsoever."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_tokens=1500,
        )
        
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.split("```")[0]
        
        return json.loads(raw.strip())
    
    def generate_report(self, analysis_results):
        report = {
            'primary_concerns': [],
            'skin_type': '',
            'recommendations': [],
            'overall_score': 0
        }
        
        weights = {
            'acne': 0.2, 'blackheads': 0.1, 'dark_spots': 0.15,
            'dryness': 0.1, 'oily': 0.1, 'pores': 0.1, 'wrinkles': 0.15
        }
        name_map = {
            'acne': 'acne', 'blackheads': 'blackheads', 'dark_spots': 'dark spots',
            'dryness': 'dryness', 'oily': 'oily', 'pores': 'pores', 'wrinkles': 'wrinkles'
        }
        
        weighted_avg_negative = 0.0
        for k, weight in weights.items():
            mapped_k = name_map[k]
            prob = analysis_results.get(mapped_k, 0.0)
            weighted_avg_negative += prob * weight
        
        score = int(100 * (1 - weighted_avg_negative))
        normal_bonus = int(analysis_results.get('normal', 0.0) * 10)
        report['overall_score'] = min(max(score + normal_bonus, 0), 100)
        
        for condition, prob in analysis_results.items():
            if prob > 0.5 and condition != 'normal':
                severity = 'High' if prob > 0.8 else 'Medium' if prob > 0.6 else 'Low'
                report['primary_concerns'].append({
                    'condition': condition.replace('_', ' ').title(),
                    'severity': severity,
                    'confidence': int(prob * 100)
                })
        
        # Determine skin type with 0.4 floor
        skin_scores = {
            'Normal': analysis_results.get('normal', 0),
            'Oily': analysis_results.get('oily', 0),
            'Dry': analysis_results.get('dryness', 0),
            'Combination': (analysis_results.get('oily', 0) + analysis_results.get('dryness', 0)) / 2
        }
        best_type = max(skin_scores, key=skin_scores.get)
        best_score = skin_scores[best_type]
        report['skin_type'] = best_type if best_score > 0.4 else 'Normal'
        
        try:
            report['ai_recommendations'] = self.generate_ai_recommendations(analysis_results, report)
            logger.info("✅ Groq AI recommendations generated successfully")
        except Exception as e:
            logger.warning(f"⚠️ Groq API failed, using fallback recommendations: {e}")
            report['ai_recommendations'] = None
            
        if report['ai_recommendations'] is None:
            recs = []
            if analysis_results.get('acne', 0) > self.thresholds.get('acne', 0.5):
                recs.append({'title': 'Acne Treatment', 'description': 'Use salicylic acid or benzoyl peroxide.'})
            if not recs:
                recs.append({'title': 'General Maintenance', 'description': 'Always use sunscreen and hydrate.'})
            report['recommendations'] = recs
        
        return report

def download_model_if_needed():
    model_path = 'models/fixed_skin_model.pth'
    thresholds_path = 'models/optimal_thresholds.json'
    os.makedirs('models', exist_ok=True)
    
    if not os.path.exists(model_path):
        print("📥 Downloading model from Hugging Face...")
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=os.environ.get('HF_REPO_ID', 'your-username/alskincare-model'),
                filename='fixed_skin_model.pth',
                local_dir='models'
            )
            print("✅ Model downloaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not download model: {e}")
    
    if not os.path.exists(thresholds_path):
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=os.environ.get('HF_REPO_ID', 'your-username/alskincare-model'),
                filename='optimal_thresholds.json',
                local_dir='models'
            )
            print("✅ Thresholds downloaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not download thresholds: {e}")

download_model_if_needed()

try:
    analyzer = SkinAnalyzerAPI()
except Exception as e:
    logger.error(f"❌ Failed to initialize Skin Analyzer: {e}")
    analyzer = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG', 'JPEG', 'PNG', 'webp', 'WEBP'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'webp'}

def validate_image_size(image_pil):
    width, height = image_pil.size
    if width < 100 or height < 100:
        return False, "Image dimensions must be at least 100x100 pixels."
    return True, ""

@app.route('/api/health', methods=['GET'])
def health_check():
    status = 'healthy' if analyzer is not None else 'unhealthy'
    return jsonify({
        'status': status,
        'message': 'Skin Analysis API is running!' if analyzer else 'Analyzer not initialized',
        'conditions': CONDITIONS
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_skin():
    if analyzer is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        file.seek(0, 0)
        
        if file_length > MAX_FILE_SIZE:
             return jsonify({'error': 'File exceeds 16MB limit.'}), 400
        
        if file and allowed_file(file.filename):
            image_pil = Image.open(file).convert('RGB')
            is_valid, err_msg = validate_image_size(image_pil)
            if not is_valid:
                return jsonify({'error': err_msg}), 400
                
            analysis_results = analyzer.analyze_skin(image_pil)
            report = analyzer.generate_report(analysis_results)
            
            response = {
                'analysis': analysis_results,
                'report': report,
                'success': True,
                'timestamp': str(np.datetime64('now'))
            }
            return jsonify(response)
        else:
            return jsonify({'error': 'Invalid file type.'}), 400
            
    except Exception as e:
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/analyze/base64', methods=['POST'])
def analyze_skin_base64():
    if analyzer is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        image_pil = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        is_valid, err_msg = validate_image_size(image_pil)
        if not is_valid:
            return jsonify({'error': err_msg}), 400
        
        analysis_results = analyzer.analyze_skin(image_pil)
        report = analyzer.generate_report(analysis_results)
        
        response = {
            'analysis': analysis_results,
            'report': report,
            'success': True,
            'timestamp': str(np.datetime64('now'))
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': 'Invalid image data', 'details': str(e)}), 400

@app.route('/api/analyze/url', methods=['POST'])
def analyze_skin_url():
    if analyzer is None:
        return jsonify({'error': 'Analyzer not initialized'}), 500
        
    try:
        data = request.get_json()
        if not data or 'image_url' not in data:
            return jsonify({'error': 'No image_url provided'}), 400
            
        url = data['image_url']
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        
        image_bytes = resp.content
        if len(image_bytes) > MAX_FILE_SIZE:
             return jsonify({'error': 'File from URL exceeds 16MB limit.'}), 400
             
        image_pil = Image.open(BytesIO(image_bytes)).convert('RGB')
        is_valid, err_msg = validate_image_size(image_pil)
        if not is_valid:
            return jsonify({'error': err_msg}), 400
            
        analysis_results = analyzer.analyze_skin(image_pil)
        report = analyzer.generate_report(analysis_results)
        
        response = {
            'analysis': analysis_results,
            'report': report,
            'success': True,
            'timestamp': str(np.datetime64('now'))
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': 'Failed to process URL', 'details': str(e)}), 400

@app.route('/api/conditions', methods=['GET'])
def get_conditions():
    return jsonify({
        'conditions': CONDITIONS,
        'count': len(CONDITIONS)
    })

@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'response': 'Invalid request data', 'success': False}), 400
            
        user_message = data.get('message', '').strip()
        skin_context = data.get('skin_context', {})
        
        if not user_message:
            return jsonify({'response': 'Please enter a message', 'success': False}), 400
        
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            return jsonify({'response': 'AI service not configured', 'success': False}), 500
        
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        
        skin_type = skin_context.get('skin_type', 'Unknown')
        overall_score = skin_context.get('overall_score', 'Unknown')
        detected_conditions = skin_context.get('detected_conditions', [])
        analysis = skin_context.get('analysis', {})
        
        system_prompt = f"""You are ALSKINCARE AI Assistant, a friendly dermatology expert chatbot.
User skin analysis results:
- Skin Type: {skin_type}
- Health Score: {overall_score}/100
- Detected Conditions: {detected_conditions}
- Condition Probabilities: {analysis}

Rules:
- Answer only skincare and dermatology questions
- Keep responses to 2-4 sentences unless more detail is needed
- Be friendly and use emojis occasionally
- For unrelated topics say: I am specialized in skincare! Ask me anything about your skin 😊
- Always suggest consulting a dermatologist for medical concerns"""

        chat_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        reply = chat_response.choices[0].message.content
        print(f"✅ Chat response generated: {reply[:50]}...")
        return jsonify({'response': reply, 'success': True})
        
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'response': f'Connection error: {str(e)}',
            'success': False
        }), 500

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 STARTING ALSKINCARE BACKEND SERVER")
    logger.info(f"🌐 Port: {PORT}")
    logger.info("=" * 60)
    
    if analyzer is None:
        logger.error("❌ CRITICAL: Analyzer failed to initialize. Check model file and dependencies.")
        exit(1)
    
    app.run(debug=False, host='0.0.0.0', port=PORT)