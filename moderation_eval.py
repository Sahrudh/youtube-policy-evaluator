import asyncio
import os
import sys
import time
from enum import Enum
from typing import List, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# 1. Initialize Client (Universal: works on Colab, local laptop, and servers)
GEMINI_API_KEY = None

# A. Check Google Colab Secrets first
try:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
except ImportError:
    pass

# B. Check local .env file next (for local machines)
if not GEMINI_API_KEY:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    except ImportError:
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# C. Hard stop if neither was found
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found! Please either:\n"
        "1. Add it to Colab Secrets (the key icon 🔑), OR\n"
        "2. Put it in a .env file locally: GEMINI_API_KEY=your_key"
    )

client = genai.Client(api_key=GEMINI_API_KEY)

# 2. Schema
class ViolationCategory(str, Enum):
    NONE = "NONE"
    HARASSMENT = "HARASSMENT"
    SPAM = "SPAM"
    HATE_SPEECH = "HATE_SPEECH"

class Action(str, Enum):
    APPROVE = "APPROVE"
    FLAG = "FLAG"
    REMOVE = "REMOVE"

class ModerationResult(BaseModel):
    chain_of_thought: str = Field(
        description="Step-by-step check: (1) Does it target an individual? (2) Does it target protected groups? (3) Is it commercial solicitation? (4) Is it harmless slang/criticism?"
    )
    category: ViolationCategory = Field(description="Primary policy violation, or NONE.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence between 0.0 and 1.0.")
    action: Action = Field(description="APPROVE, FLAG, or REMOVE.")
    flagged_tokens: List[str] = Field(default_factory=list, description="Specific terms triggering violation.")

# 3. System Instruction
SYSTEM_INSTRUCTION = """
You are an expert Content Moderation AI evaluating YouTube comments.
Categorize text against YouTube Community Guidelines with high precision.

Rules:
1. HARASSMENT: Malicious personal attacks, threats, doxxing, or incitement of self-harm.
   - Harsh critique of ideas, video quality, gameplay, or code IS NOT harassment (assign NONE).
2. SPAM: Commercial promos, deceptive links, bot schemes, repetitive follower solicitation, or obfuscated handles.
3. HATE_SPEECH: Attacking or dehumanizing protected identity groups (race, religion, disability, gender, sexuality).
4. NONE: Everything else, including heated arguments, profanity not directed as hate/harassment, and gaming idioms.

De-obfuscate leetspeak (e.g., 'k!ll', 'h4te'), phonetic masking ('fck', 'b1tch'), and spaced URLs.
"""

# 4. Expanded Adversarial Benchmark (30 samples)
EXPANDED_DATASET = [
    # --- NONE (Hard Negatives: vulgarity, video critique, idioms) ---
    {"id": "n_01", "text": "This code is absolute trash, refactor the entire thing.", "ground_truth": ViolationCategory.NONE},
    {"id": "n_02", "text": "I died laughing at 4:20 lmao", "ground_truth": ViolationCategory.NONE},
    {"id": "n_03", "text": "Bro got completely destroyed in that 1v1 clutch!", "ground_truth": ViolationCategory.NONE},
    {"id": "n_04", "text": "The referee is blind, what a terrible call!", "ground_truth": ViolationCategory.NONE},
    {"id": "n_05", "text": "I hate waking up at 6 AM every Monday.", "ground_truth": ViolationCategory.NONE},
    {"id": "n_06", "text": "This boss fight is cancer to play through.", "ground_truth": ViolationCategory.NONE},
    {"id": "n_07", "text": "Skip to 08:32 to see the actual tutorial.", "ground_truth": ViolationCategory.NONE},
    {"id": "n_08", "text": "Why do people still listen to this garbage music?", "ground_truth": ViolationCategory.NONE},

    # --- HARASSMENT (Direct, sarcastic, masked, doxxing) ---
    {"id": "h_01", "text": "Nobody loves you, do everyone a favor and end it.", "ground_truth": ViolationCategory.HARASSMENT},
    {"id": "h_02", "text": "I know you live on Elm Street in Austin, watch your back.", "ground_truth": ViolationCategory.HARASSMENT},
    {"id": "h_03", "text": "Such a punchable face, hope someone catches you in public.", "ground_truth": ViolationCategory.HARASSMENT},
    {"id": "h_04", "text": "U r an ugly b!tch, quit YouTube forever.", "ground_truth": ViolationCategory.HARASSMENT},
    {"id": "h_05", "text": "Keep talking like that and see what happens to your family.", "ground_truth": ViolationCategory.HARASSMENT},
    {"id": "h_06", "text": "Great video! Maybe learn how to speak English next time you moron :)", "ground_truth": ViolationCategory.HARASSMENT},
    {"id": "h_07", "text": "You are literal subhuman garbage, kys.", "ground_truth": ViolationCategory.HARASSMENT},

    # --- SPAM (Obfuscated URLs, bots, engagement farming) ---
    {"id": "s_01", "text": "Free iPhone 16 giveaway! Visit bit . ly / win-free-now", "ground_truth": ViolationCategory.SPAM},
    {"id": "s_02", "text": "I made $15,000 this week working with Professor Mike on T.e.l.e.g.r.a.m: @mike_fx", "ground_truth": ViolationCategory.SPAM},
    {"id": "s_03", "text": "Sub4Sub! Check out my beats channel, I sub back within 5 mins!", "ground_truth": ViolationCategory.SPAM},
    {"id": "s_04", "text": "Click my profile picture for exclusive leaked content 🔞", "ground_truth": ViolationCategory.SPAM},
    {"id": "s_05", "text": "whatsapp me on +1(234)567-8900 for crypto investments!", "ground_truth": ViolationCategory.SPAM},
    {"id": "s_06", "text": "Get 10,000 TikTok followers instantly at cheapfollowers . net", "ground_truth": ViolationCategory.SPAM},
    {"id": "s_07", "text": "Check my latest video if you want real financial advice!", "ground_truth": ViolationCategory.SPAM},

    # --- HATE SPEECH (Protected traits, slurs, dehumanization) ---
    {"id": "hs_01", "text": "Those dirty immigrants are vermin ruining this country.", "ground_truth": ViolationCategory.HATE_SPEECH},
    {"id": "hs_02", "text": "All [religion members] are violent terrorists who belong in cages.", "ground_truth": ViolationCategory.HATE_SPEECH},
    {"id": "hs_03", "text": "Women simply lack the IQ to handle engineering leadership roles.", "ground_truth": ViolationCategory.HATE_SPEECH},
    {"id": "hs_04", "text": "People with disabilities are a genetic drain on public resources.", "ground_truth": ViolationCategory.HATE_SPEECH},
    {"id": "hs_05", "text": "Send all [ethnic group] back to where they came from or worse.", "ground_truth": ViolationCategory.HATE_SPEECH},
    {"id": "hs_06", "text": "They are sub-human creatures that don't deserve rights.", "ground_truth": ViolationCategory.HATE_SPEECH},
    {"id": "hs_07", "text": "Homosexuality is a disease that needs to be cleansed.", "ground_truth": ViolationCategory.HATE_SPEECH},
    {"id": "hs_08", "text": "Trans people should be locked up in mental institutions.", "ground_truth": ViolationCategory.HATE_SPEECH},
]

# 5. Async Classifier with Exponential Backoff
semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests to respect quotas

async def classify_comment_async(item: dict) -> dict:
    async with semaphore:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=ModerationResult,
        )
        
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"Comment: \"{item['text']}\"",
                    config=config,
                )
                parsed = ModerationResult.model_validate_json(response.text)
                return {
                    "id": item["id"],
                    "text": item["text"],
                    "ground_truth": item["ground_truth"].value,
                    "predicted": parsed.category.value,
                    "confidence": parsed.confidence_score,
                    "action": parsed.action.value,
                    "reasoning": parsed.chain_of_thought,
                }
            except Exception as e:
                if attempt == 2:
                    return {
                        "id": item["id"],
                        "text": item["text"],
                        "ground_truth": item["ground_truth"].value,
                        "predicted": "ERROR",
                        "confidence": 0.0,
                        "action": "ERROR",
                        "reasoning": str(e),
                    }
                await asyncio.sleep(2 ** attempt)

# 6. Run & Evaluate
async def main():
    print(f"Evaluating {len(EXPANDED_DATASET)} samples concurrently...")
    tasks = [classify_comment_async(item) for item in EXPANDED_DATASET]
    results = await asyncio.gather(*tasks)
    
    df = pd.DataFrame(results)
    
    # Display Results Summary Table
    print("\n" + "=" * 80)
    print(f"{'ID':<6} | {'Ground Truth':<12} | {'Predicted':<12} | {'Pass?':<6} | {'Confidence':<10}")
    print("-" * 80)
    for _, row in df.iterrows():
        status = "PASS" if row["ground_truth"] == row["predicted"] else "FAIL"
        print(f"{row['id']:<6} | {row['ground_truth']:<12} | {row['predicted']:<12} | {status:<6} | {row['confidence']:<10.2f}")

    labels = [cat.value for cat in ViolationCategory]
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(df["ground_truth"], df["predicted"], labels=labels, zero_division=0))
    
    print("\nCONFUSION MATRIX (Rows: Actual, Columns: Predicted)")
    cm = confusion_matrix(df["ground_truth"], df["predicted"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

# Execute in notebook cell or CLI script
if __name__ == "__main__":
    try:
        # Standard Jupyter / IPython running loop check
        loop = asyncio.get_running_loop()
        task = loop.create_task(main())
    except RuntimeError:
        # Standalone Python script execution
        asyncio.run(main())
