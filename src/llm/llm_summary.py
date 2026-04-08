"""
Enhanced LLM-based summary generation for Multi-Stock support.
Provides detailed financial analysis using flan-t5 or high-quality templates.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.utils.helpers import get_device

# Tip: If your PC has 8GB+ RAM, change "small" to "base" for even better text
MODEL_NAME = "google/flan-t5-small" 

class LLMSummarizer:
    def __init__(self):
        self.device = get_device()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            self.model = self.model.to(torch.device("cpu"))
            self.model.eval()
        except Exception:
            self.model = None
            self.tokenizer = None
    
    def generate_summary(self, context_dict):
        if self.model is None:
            return self._generate_template_summary(context_dict)
        
        try:
            prompt = self._construct_prompt(context_dict)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                # Increased max_new_tokens to 120 for longer explanations
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=120, 
                    num_beams=5, 
                    repetition_penalty=2.5,
                    length_penalty=1.5
                )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # If the AI is still being too lazy/short (under 40 chars), use the detailed template
            if len(summary) < 40:
                return self._generate_template_summary(context_dict)
            
            return summary
        except Exception:
            return self._generate_template_summary(context_dict)

    def _construct_prompt(self, context_dict):
        stock_name = context_dict.get("ticker_name", "the selected stock")
        predicted_price = context_dict.get("predicted_price", 0)
        last_price = context_dict.get("last_price", 0)
        change_pct = context_dict.get("predicted_change_pct", 0)
        sentiment = context_dict.get("sentiment_summary", "neutral")
        
        direction = "upward" if change_pct > 0 else "downward"
        
        # New prompt format designed to get longer responses
        return f"Task: Provide a detailed financial market analysis for {stock_name}. " \
               f"Data: Current price is ₹{last_price:.2f}. The AI model predicts a price of ₹{predicted_price:.2f} " \
               f"which is a {abs(change_pct):.2f}% {direction} move. The news sentiment is {sentiment}. " \
               f"Analysis: "

    def _generate_template_summary(self, context_dict):
        """Detailed fallback template when the LLM output is too short."""
        stock_name = context_dict.get("ticker_name", "the selected stock")
        predicted_price = context_dict.get("predicted_price", 0)
        last_price = context_dict.get("last_price", 0)
        change_pct = context_dict.get("predicted_change_pct", 0)
        sentiment = context_dict.get("sentiment_summary", "neutral")
        model_name = context_dict.get("model_name", "LSTM")
        
        direction = "bullish (upward)" if change_pct > 0 else "bearish (downward)"
        magnitude = "significant" if abs(change_pct) > 2 else "moderate" if abs(change_pct) > 1 else "mild"

        if abs(change_pct) > 0.05:
            return (
                f"📈 **Technical Market Insight for {stock_name}:**\n\n"
                f"The {model_name} architecture has identified a **{magnitude} {direction} trend**. "
                f"The model anticipates the price moving from the current ₹{last_price:.2f} towards a target of ₹{predicted_price:.2f} "
                f"in the next session. This expected {abs(change_pct):.2f}% shift is reinforced by a **{sentiment}** news cycle, "
                f"suggesting that technical momentum and public sentiment are currently aligned. Investors should watch for "
                f"volatility near the target price."
            )
        else:
            return (
                f"📊 **Consolidation Analysis for {stock_name}:**\n\n"
                f"Current technical indicators suggest that **{stock_name}** is entering a consolidation phase. "
                f"The {model_name} predicts minimal price variance (around {abs(change_pct):.2f}%), indicating that "
                f"buying and selling pressures are currently neutralized. With news sentiment remaining **{sentiment}**, "
                f"the stock is expected to hold its current support levels near ₹{last_price:.2f} without a major breakout."
            )

def generate_summary(context_dict):
    summarizer = LLMSummarizer()
    return summarizer.generate_summary(context_dict)