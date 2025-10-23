"""
Flask Backend for Voice Payment Recognition using OpenAI Whisper + Razorpay

Installation:
    pip install openai-whisper flask flask-cors razorpay

Usage:
    1. Set your Razorpay credentials in the code below
    2. python backend.py
    3. Open the HTML file in browser
"""

from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
import whisper
import razorpay
import tempfile
import os
import re

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for HTML to communicate with this server

# ============= RAZORPAY CONFIGURATION =============
# IMPORTANT: Replace these with your actual Razorpay credentials
# Get them from: https://dashboard.razorpay.com/app/keys
RAZORPAY_KEY_ID = os.getenv('RAZORPAY_KEY_ID')          #linve mode key id
RAZORPAY_KEY_SECRET = os.getenv('RAZORPAY_KEY_SECRET')  #live mode secret id

# For testing, use test keys:
# RAZORPAY_KEY_ID = "rzp_test_YOUR_TEST_KEY"
# RAZORPAY_KEY_SECRET = "YOUR_TEST_SECRET"

# Initialize Razorpay client
razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
# ==================================================

# Load Whisper model once at startup
print("Loading OpenAI Whisper model...")
model = whisper.load_model("base", device="cpu")  # Force CPU to avoid GPU compatibility issues
print("Model loaded successfully!")

def extract_amount(text):
    """
    Extract numeric amount from transcribed text
    
    Args:
        text: Transcribed text
        
    Returns:
        Extracted amount as integer or None
    """
    # First, try to find direct digits
    digits = re.findall(r'\d+', text)
    if digits:
        return int(digits[0])
    
    # Word to number conversion
    words_to_numbers = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
        'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000
    }
    
    total = 0
    current = 0
    words = text.lower().split()
    
    for word in words:
        # Remove punctuation
        word = re.sub(r'[^\w\s]', '', word)
        
        if word in words_to_numbers:
            num = words_to_numbers[word]
            
            if num == 100:
                current = current * 100 if current else 100
            elif num == 1000:
                current = (current if current else 1) * 1000
                total += current
                current = 0
            else:
                current += num
    
    total += current
    return total if total > 0 else None

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    API endpoint to receive audio and return transcribed amount
    
    Expects:
        Audio file in request.files['audio']
        
    Returns:
        JSON with amount and transcribed text
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_path = temp_audio.name
        
        try:
            # Transcribe using Whisper
            print("Transcribing audio...")
            result = model.transcribe(temp_path, language="en")
            transcribed_text = result["text"]
            print(f"Transcribed: {transcribed_text}")
            
            # Extract amount
            amount = extract_amount(transcribed_text)
            
            response = {
                'success': True,
                'text': transcribed_text,
                'amount': amount
            }
            
            return jsonify(response)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/create-order', methods=['POST'])
def create_order():
    """
    Create a Razorpay order
    
    Expects:
        JSON with 'amount' (in rupees), 'upi_id' (optional)
        
    Returns:
        JSON with order_id, amount, and razorpay_key_id
    """
    try:
        data = request.json
        amount = data.get('amount')  # Amount in rupees
        upi_id = data.get('upi_id', 'Not provided')
        
        if not amount or amount <= 0:
            return jsonify({'error': 'Invalid amount'}), 400
        
        # Create Razorpay order
        order_data = {
            'amount': int(amount) * 100,  # Convert rupees to paise
            'currency': 'INR',
            'payment_capture': 1,  # Auto-capture payment
            'notes': {
                'upi_id': upi_id,
                'payment_method': 'voice_payment'
            }
        }
        
        order = razorpay_client.order.create(data=order_data)
        
        print(f"Order created: {order['id']} for amount: ₹{amount}")
        
        return jsonify({
            'success': True,
            'order_id': order['id'],
            'amount': order['amount'],
            'currency': order['currency'],
            'razorpay_key_id': RAZORPAY_KEY_ID  # Send key to frontend
        })
    
    except Exception as e:
        print(f"Error creating order: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/verify-payment', methods=['POST'])
def verify_payment():
    """
    Verify Razorpay payment signature
    
    Expects:
        JSON with razorpay_order_id, razorpay_payment_id, razorpay_signature
        
    Returns:
        JSON with success status and payment details
    """
    try:
        data = request.json
        
        razorpay_order_id = data.get('razorpay_order_id')
        razorpay_payment_id = data.get('razorpay_payment_id')
        razorpay_signature = data.get('razorpay_signature')
        
        if not all([razorpay_order_id, razorpay_payment_id, razorpay_signature]):
            return jsonify({'error': 'Missing payment details'}), 400
        
        # Verify signature
        params_dict = {
            'razorpay_order_id': razorpay_order_id,
            'razorpay_payment_id': razorpay_payment_id,
            'razorpay_signature': razorpay_signature
        }
        
        try:
            razorpay_client.utility.verify_payment_signature(params_dict)
            
            # Fetch payment details
            payment = razorpay_client.payment.fetch(razorpay_payment_id)
            
            print(f"Payment verified successfully: {razorpay_payment_id}")
            print(f"Amount: ₹{payment['amount']/100}, Status: {payment['status']}")
            
            return jsonify({
                'success': True,
                'message': 'Payment verified successfully',
                'payment_id': razorpay_payment_id,
                'amount': payment['amount'] / 100,  # Convert paise to rupees
                'status': payment['status']
            })
        
        except razorpay.errors.SignatureVerificationError:
            print("Payment signature verification failed!")
            return jsonify({
                'success': False,
                'message': 'Payment verification failed - Invalid signature'
            }), 400
    
    except Exception as e:
        print(f"Error verifying payment: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get-razorpay-key', methods=['GET'])
def get_razorpay_key():
    """Return Razorpay Key ID for frontend"""
    return jsonify({'key_id': RAZORPAY_KEY_ID})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'whisper_model': 'whisper-base',
        'razorpay_configured': bool(RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET)
    })

if __name__ == '__main__':
    # Validation check
    if "YOUR_KEY" in RAZORPAY_KEY_ID or "YOUR_SECRET" in RAZORPAY_KEY_SECRET:
        print("\n" + "="*60)
        print("⚠️  WARNING: Please configure your Razorpay credentials!")
        print("   Edit RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in this file")
        print("   Get credentials from: https://dashboard.razorpay.com/app/keys")
        print("="*60 + "\n")
    
    print("\n" + "="*60)
    print("Flask Server Running on http://localhost:5000")
    print("Whisper model ready for speech recognition")
    print("Razorpay payment gateway integrated")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)