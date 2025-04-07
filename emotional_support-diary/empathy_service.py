'''no longer needed 
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

class EmpathyService:
    def __init__(self):
        # Load BlenderBot-400M-Distill
        model_name = "facebook/blenderbot-400M-distill"
        self.tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

    def generate_empathetic_response(self, journal_text: str, detected_emotion: str = None) -> str:
        prompt = f"You are an empathetic AI. Respond to the following journal entry with kindness and support:\n\nJournal Entry: {journal_text}\n\nEmotion: {detected_emotion or 'Neutral'}\n\nResponse:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        reply_ids = self.model.generate(**inputs)
        response = self.tokenizer.decode(reply_ids[0], skip_special_tokens=True)

        # Extract only the meaningful part of the response
        if "Response:" in response:
            return response.split("Response:", 1)[1].strip()
        return response
    
'''  