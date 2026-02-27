import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import datetime as dt
from sklearn.preprocessing import MinMaxScaler

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# ===============================================================
# Call DeepSeek-R1 (HF) – NON-COT numeric interface
# ===============================================================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
)
model.to(device)
model.eval()


def call_deepseek(prompt: str, max_new_tokens: int = 256) -> str:
    """
    Run DeepSeek-R1-0528-Qwen3-8B via HF transformers, strip <think>...</think>,
    and return plain text.
    """

    # Use chat template (Qwen-style)
    messages = [
        {"role": "user", "content": prompt}
    ]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # deterministic for this numeric use-case
            pad_token_id=tokenizer.eos_token_id,
        )

    # Take only the generated continuation (exclude the prompt tokens)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # DeepSeek-style reasoning often wrapped in <think>...</think>, so strip it
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


# ===============================================================
# Prompt generator (non-COT, numeric-only)
# ===============================================================
def generate_prompt(selected_segment, segment_length=300, prediction_length=20):
    segment_str = np.array2string(
        np.array(selected_segment),
        precision=7,
        separator=', ',
        max_line_width=80,
        suppress_small=False,
        prefix='      '
    )

    # Tight numeric-only prompt like in your Ollama script
    #prompt = (
    #    f"These are normalized stock prices (between 0 and 1) for the first {segment_length} days:\n"
    #    f"{segment_str}\n\n"
    #    f"Predict the stock price for the next {prediction_length} days as "
    #    f"{prediction_length} normalized numbers between 0 and 1.\n"
    #    f"Output ONLY the {prediction_length} numbers, separated by spaces, "
    #    f"with no explanation or extra text."
    #)
    #return prompt
    prompt = (
        f"this is the plot of a stock over the first {segment_length} days, "
        f"and these are the time-series values:\n"
        f"      {segment_str}. "
        f"Considering both the plot and time-series values, predict the stock price "
        f"for the next {prediction_length} days approximately."
    )
    return prompt



# ===============================================================
# Load CAC40 dataset
# ===============================================================
df = pd.read_csv("data/preprocessed_CAC40.csv", parse_dates=["Date"])
df = df.drop(columns=["Unnamed: 0"])

def extract_company_data(company, start, end):
    tmp = df[df["Name"] == company]
    return tmp[(tmp["Date"] > start) & (tmp["Date"] < end)]


# Use the second company
company_name = df["Name"].unique()[6] #BNP Paribas
print("company_name=", company_name)
start_date = dt.datetime(2014, 1, 1)
end_date   = dt.datetime(2020, 1, 1)

specific_df = extract_company_data(company_name, start_date, end_date)
prices = specific_df.reset_index()["Closing_Price"]

# Normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices.values.reshape(-1, 1))
stock_list = scaled.reshape(-1,)


# ===============================================================
# Parameters
# ===============================================================
segment_length = 100
prediction_length = 5
step_forward = 100

mse_errors = []


# ===============================================================
# Main NON-COT loop
# ===============================================================
for i in range(0, len(stock_list) - (segment_length + prediction_length), step_forward):

    print("\nIndex:", i)

    segment = stock_list[i : i + segment_length]
    target  = stock_list[i + segment_length : i + segment_length + prediction_length]

    prompt = generate_prompt(segment, segment_length, prediction_length)

    # Send to DeepSeek via HF
    raw_output = call_deepseek(prompt)
    print("LLM raw output:", raw_output)
    print("Ground Truth:", target)
    
    # ===============================================================
    # Extract numeric predictions
    # ===============================================================
    tokens = raw_output.replace("\n", " ").split()
    numeric_vals = []

    for tok in tokens:
        clean = tok.strip().replace(",", "")
        # crude float check: optional single '.'
        if clean.replace(".", "", 1).isdigit():
            try:
                numeric_vals.append(float(clean))
            except ValueError:
                continue

    if len(numeric_vals) == 0:
        print("No numeric values detected.")
        continue

    # keep only normalized numbers <1
    numeric_vals = [x for x in numeric_vals if x < 1]
    #numeric_vals = numeric_vals[:prediction_length]

    # if fewer numbers than expected, we still use them (partial MSE) (take at most prediction_length values, but allow fewer)
    pred = np.array(numeric_vals[:prediction_length])
    truth = np.array(target[:len(pred)])
    
    print("Parsed values:", pred)
    
    if len(pred) == 0:
        print("Zero usable numeric predictions.")
        continue


    print("Parsed values:", pred)

    #if len(numeric_vals) < prediction_length:
    #    print("Insufficient values.")
    #    continue

    if len(pred) == 0:
        print("Zero usable numeric predictions.")
        continue



    # ===============================================================
    # Partial MSE
    # ===============================================================
    #pred = np.array(numeric_vals)
    #truth = target[:len(pred)]

    mse = np.mean((pred - truth)**2)
    mse_errors.append(mse)

    #print("MSE:", mse)
    print(f"MSE (using {len(pred)} predictions):", mse)

# ===============================================================
# Summary
# ===============================================================
print("\nAll MSE errors:", mse_errors)

if len(mse_errors) == 0:
    print("No MSE values were computed. Nothing to average.")
else:
    avg_mse = np.mean(mse_errors)
    print("\n====================================")
    print(f"Total evaluated windows: {len(mse_errors)}")
    print(f"Average MSE: {avg_mse}")
    print("====================================")
