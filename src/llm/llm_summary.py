"""
LLM-based summary generation using flan-t5-small for prediction explanations.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_device


MODEL_NAME = "google/flan-t5-small"


class LLMSummarizer:
    """LLM-based summarizer for prediction explanations."""
    
    def __init__(self):
        """Initialize flan-t5-small model."""
        self.device = get_device()
        print(f"Loading {MODEL_NAME} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            
            # Move to device (flan-t5-small works on CPU, but can use GPU/MPS if available)
            if self.device.type == "cuda":
                self.model = self.model.to(self.device)
            elif self.device.type == "mps":
                # T5 models may have MPS issues, use CPU for now
                self.model = self.model.to(torch.device("cpu"))
                self.device = torch.device("cpu")
            else:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print(f"{MODEL_NAME} loaded successfully")
        except Exception as e:
            print(f"Error loading {MODEL_NAME}: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate_summary(self, context_dict):
        """
        Generate a 2-3 sentence summary explaining the prediction.
        
        Args:
            context_dict (dict): Context dictionary with keys:
                - predicted_price: Predicted price
                - last_price: Last known price
                - predicted_change_pct: Predicted change percentage
                - sentiment_summary: Summary of sentiment (e.g., "mostly positive")
                - model_name: Model used for prediction
        
        Returns:
            str: Generated summary text
        """
        if self.model is None:
            # Fallback to template-based summary
            return self._generate_template_summary(context_dict)
        
        try:
            # Construct prompt
            prompt = self._construct_prompt(context_dict)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get input length to exclude from generation
            input_length = inputs["input_ids"].shape[1]
            
            # Generate (only new tokens, excluding input)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=80,  # Only generate new tokens (T5 style)
                    num_beams=3,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.7,  # Add some randomness for better generation
                    top_p=0.9,
                    no_repeat_ngram_size=2  # Avoid repetition
                )
            
            # Decode only the newly generated tokens (skip input tokens)
            generated_tokens = outputs[0][input_length:]  # Extract only new tokens
            summary = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up the summary
            summary = summary.strip()
            
            # Remove common prefixes/artifacts that T5 might add
            summary = summary.replace("Summary:", "").strip()
            summary = summary.replace("summary:", "").strip()
            summary = summary.replace("Summarize:", "").strip()
            
            # Check if output is meaningful (not just repeating the prompt or too short)
            # Don't reject just because it starts with "the" - that's normal
            is_meaningful = (
                len(summary) >= 30 and  # At least 30 characters
                not summary.lower().startswith(("summarize", "explain the", "explain:")) and  # Not task repetition
                len(summary.split()) >= 5  # At least 5 words
            )
            
            # If summary is not meaningful, use fallback
            if not is_meaningful:
                print(f"LLM output too short/meaningless: '{summary[:50]}...' Using template fallback.")
                return self._generate_template_summary(context_dict)
            
            return summary
        
        except Exception as e:
            print(f"Error generating summary: {e}")
            return self._generate_template_summary(context_dict)
    
    def _construct_prompt(self, context_dict):
        """Construct prompt for the LLM."""
        predicted_price = context_dict.get("predicted_price", 0)
        last_price = context_dict.get("last_price", 0)
        change_pct = context_dict.get("predicted_change_pct", 0)
        sentiment = context_dict.get("sentiment_summary", "neutral")
        model_name = context_dict.get("model_name", "model")
        
        direction = "increase" if change_pct > 0 else "decrease"
        magnitude = "significantly" if abs(change_pct) > 2 else "moderately" if abs(change_pct) > 1 else "slightly"
        
        # For T5 models (flan-t5-small), use instruction-based prompts
        # flan-t5 works best with clear instructions
        if abs(change_pct) > 0.01:  # If there's a meaningful change
            prompt = f"""In 2 sentences, explain the stock prediction for Adani Green Energy:
            Current price: ₹{last_price:.2f}
            Predicted price: ₹{predicted_price:.2f} ({abs(change_pct):.2f}% {direction})
            News sentiment: {sentiment}
            Model: {model_name}
            
            Explanation:"""
        else:  # If no significant change
            prompt = f"""In 2 sentences, explain the current market situation for Adani Green Energy:
            Current price: ₹{last_price:.2f}
            Change: {abs(change_pct):.2f}%
            News sentiment: {sentiment}
            
            Explanation:"""
        
        return prompt
    
    def _generate_template_summary(self, context_dict):
        """Generate an intelligent summary with insights based on predictions.
        
        This method creates comprehensive insights that combine:
        - Price predictions and trends
        - Sentiment analysis
        - Market context
        - Actionable insights
        """
        predicted_price = context_dict.get("predicted_price", 0)
        last_price = context_dict.get("last_price", 0)
        change_pct = context_dict.get("predicted_change_pct", 0)
        sentiment = context_dict.get("sentiment_summary", "neutral")
        model_name = context_dict.get("model_name", "unknown")
        
        # Determine direction and magnitude
        direction = "increase" if change_pct > 0 else "decrease" if change_pct < 0 else "stability"
        
        # Categorize magnitude
        if abs(change_pct) > 5:
            magnitude = "substantial"
            trend_strength = "strong"
        elif abs(change_pct) > 2:
            magnitude = "significant"
            trend_strength = "moderate to strong"
        elif abs(change_pct) > 1:
            magnitude = "moderate"
            trend_strength = "moderate"
        elif abs(change_pct) > 0.1:
            magnitude = "slight"
            trend_strength = "mild"
        else:
            magnitude = "minimal"
            trend_strength = "weak"
        
        # Determine sentiment alignment
        sentiment_alignment = ""
        if sentiment == "mostly positive" and change_pct > 0:
            sentiment_alignment = "This aligns well with positive news sentiment, suggesting investor confidence."
        elif sentiment == "mostly positive" and change_pct < 0:
            sentiment_alignment = "Despite positive news, the model predicts a decline, indicating potential market correction or overvaluation concerns."
        elif sentiment == "mostly negative" and change_pct < 0:
            sentiment_alignment = "This aligns with negative sentiment, reflecting market concerns."
        elif sentiment == "mostly negative" and change_pct > 0:
            sentiment_alignment = "Despite negative news, the model predicts growth, suggesting resilience or oversold conditions."
        else:
            sentiment_alignment = "Neutral sentiment suggests balanced market conditions."
        
        # Generate comprehensive insights
        if abs(change_pct) > 0.1:
            summary = (
                f"📈 **Market Prediction Analysis:**\n\n"
                f"The {model_name} model forecasts a {magnitude} {direction} for Adani Green Energy stock, "
                f"predicting the price to move from ₹{last_price:.2f} to ₹{predicted_price:.2f}, "
                f"representing a {abs(change_pct):.2f}% change. This indicates a {trend_strength} price trend. "
                f"{sentiment_alignment} "
                f"Investors should monitor market conditions closely, as this prediction suggests "
                f"{'potential upside' if change_pct > 0 else 'possible downside'} based on current technical and sentiment indicators."
            )
        else:
            summary = (
                f"📊 **Market Stability Analysis:**\n\n"
                f"Current market analysis for Adani Green Energy indicates price stability around ₹{last_price:.2f}, "
                f"with minimal expected movement ({abs(change_pct):.2f}% change). The {model_name} suggests "
                f"consolidation phase, indicating balanced supply and demand. Recent news sentiment is {sentiment}, "
                f"supporting the stable outlook. This suggests the stock may be in a consolidation phase, "
                f"with limited volatility expected in the near term."
            )
        
        return summary


def generate_summary(context_dict):
    """
    Generate summary using LLM summarizer.
    
    Args:
        context_dict (dict): Context dictionary for summary generation
    
    Returns:
        str: Generated summary
    """
    summarizer = LLMSummarizer()
    return summarizer.generate_summary(context_dict)


if __name__ == "__main__":
    # Test summary generation
    print("Testing LLM summary generation...")
    
    test_context = {
        "predicted_price": 1250.50,
        "last_price": 1200.00,
        "predicted_change_pct": 4.21,
        "sentiment_summary": "mostly positive",
        "model_name": "LSTM"
    }
    
    summarizer = LLMSummarizer()
    summary = summarizer.generate_summary(test_context)
    print(f"\nGenerated summary:\n{summary}")

