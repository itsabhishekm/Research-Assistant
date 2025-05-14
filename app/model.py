from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LocalGPTModel:
    def __init__(self):
        self.model_name = "EleutherAI/gpt-j-6B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_answer(self, question: str, context: str, max_new_tokens: int = 128) -> str:
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        self.tokenizer.truncation_side = "left"

        max_input_len = self.model.config.n_positions - max_new_tokens

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_ids = outputs[0][prompt_len:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return answer.strip()
