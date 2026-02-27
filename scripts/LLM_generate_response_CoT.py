import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import datetime as dt
from sklearn.preprocessing import MinMaxScaler

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ===============================================================
# Prompt generator (COT)
# ===============================================================
def generate_prompt_COT(selected_segment, segment_length=300, prediction_length=20):
    segment_str = np.array2string(
        np.array(selected_segment),
        precision=7,
        separator=', ',
        max_line_width=80,
        suppress_small=False,
        prefix='      '
    )

    prompt = (
        f"this is the plot of a stock over the first {segment_length} days, "
        f"and these are the time-series values:\n"
        f"      {segment_str}.\n"
        f"Considering both the plot and the time-series values, examine whether the trend "
        f"is increasing, decreasing, stabilizing, or fluctuating. "
        f"Predict the stock price for the next {prediction_length} days approximately."
    )
    return prompt


# ===============================================================
# Load CAC40 data
# ===============================================================
df = pd.read_csv("preprocessed_CAC40.csv", parse_dates=["Date"])
df = df.drop(columns=["Unnamed: 0"])

def extract_company_data(company, start, end):
    tmp = df[df["Name"] == company]
    tmp = tmp[(tmp["Date"] > start) & (tmp["Date"] < end)]
    return tmp


# choose the 2nd stock
company_name = df["Name"].unique()[1]

start_date = dt.datetime(2014, 1, 1)
end_date   = dt.datetime(2020, 1, 1)

specific_df = extract_company_data(company_name, start_date, end_date)
prices = specific_df.reset_index()["Closing_Price"]

# normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices.values.reshape(-1, 1))
stock_list = scaled.reshape(-1,)


# ===============================================================
# Load T5 LLM
# ===============================================================
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base").cuda()


# ===============================================================
# Parameters
# ===============================================================
segment_length = 100
prediction_length = 5
step_forward = 100

mse_errors = []


# ===============================================================
# Main CoT LOOP
# ===============================================================
for i in range(0, len(stock_list) - (segment_length + prediction_length), step_forward):

    print("\nIndex:", i)

    segment = stock_list[i : i + segment_length]
    target  = stock_list[i + segment_length : i + segment_length + prediction_length]

    prompt = generate_prompt_COT(segment, segment_length, prediction_length)

    # LLM forward
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=512
    )

    raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("LLM raw output:", raw_output)

    # ===============================================================
    # Parse numeric predictions
    # ===============================================================
    tokens = raw_output.replace("\n", " ").split()
    numeric_vals = []

    for tok in tokens:
        clean = tok.strip().replace(",", "")
        if clean.replace(".", "", 1).isdigit():
            numeric_vals.append(float(clean))

    if len(numeric_vals) == 0:
        print("No numeric values detected.")
        continue

    # keep only normalized values under 1
    numeric_vals = [x for x in numeric_vals if x < 1]
    numeric_vals = numeric_vals[:prediction_length]

    print("Parsed values:", numeric_vals)

    if len(numeric_vals) < prediction_length:
        print("Insufficient values.")
        continue

    # ===============================================================
    # MSE
    # ===============================================================
    pred = np.array(numeric_vals)
    truth = target[:len(pred)]

    mse = np.mean((pred - truth)**2)
    mse_errors.append(mse)

    print("MSE:", mse)


# ===============================================================
# Summary
# ===============================================================
print("\nAll MSE errors:", mse_errors)
