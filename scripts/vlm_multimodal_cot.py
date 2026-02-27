import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import warnings

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

warnings.filterwarnings("ignore")

# -----------------------------
# Plotting
# -----------------------------
def load_image(selected_segment, save_path="segment_plot.png"):
    plt.figure()
    plt.plot(selected_segment, linewidth=2.5)
    plt.grid(True)
    plt.title('Closing Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.savefig(save_path)
    plt.close()
    return save_path

# -----------------------------
# Prompt generator (COT GPT-style)
# -----------------------------
def generate_prompt_COT_p_gpt(selected_segment, segment_length=100, prediction_length=5):
    segment_str = np.array2string(
        np.array(selected_segment),
        precision=7,
        separator=', ',
        max_line_width=80,
        suppress_small=False,
        prefix='      '
    )

    return (
        f"Here is a plot showing a stock’s price over the first {segment_length} days, "
        f"along with the corresponding time-series data:\n"
        f"{segment_str}\n"
        f"Based on the visible trend and the shape of the plot, estimate the next {prediction_length} values.\n"
        f"This is a hypothetical projection based on the plot and series values only.\n"
        f"Only output the next {prediction_length} predicted prices as a Python list of floats."
    )

# -----------------------------
# Data extraction helper
# -----------------------------
def load_stock_segments(csv_path, company, start, end):
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df[df['Name'] == company]
    df = df[(df['Date'] > start) & (df['Date'] < end)]
    prices = df.reset_index()['Closing_Price']

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(np.array(prices).reshape(-1, 1))
    return scaled.reshape(-1,), scaler

# -----------------------------
# DeepSeek-VL2-tiny model prep
# -----------------------------
model_path = "deepseek-ai/deepseek-vl2-tiny"

vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# -----------------------------
# VLM call using DeepSeek-VL2
# -----------------------------
def get_response_from_vlm(prompt: str, image_path: str) -> str:
    """
    image_path: path to the saved plot image (PNG)
    """

    # Build DeepSeek-style conversation
    # System instructions can go into the user content or system_prompt; here we bake into content.
    user_content = (
        "<image>\n"
        "You are an agent that predicts stock prices from plots and time series.\n"  #optional
        + prompt
    )

    conversation = [
        {
            "role": "<|User|>",
            "content": user_content,
            "images": [image_path],
        },
        {
            "role": "<|Assistant|>",
            "content": "",
        },
    ]

    # Load PIL images from the conversation spec
    pil_images = load_pil_images(conversation)

    # Prepare model inputs
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)

    # Run image encoder + language model
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
    )

    # Decode full sequence then strip to assistant part
    decoded = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)

    # Heuristic: take substring after "<|Assistant|>:"
    if "<|Assistant|>:" in decoded:
        decoded = decoded.split("<|Assistant|>:", 1)[1].strip()

    return decoded.strip()

# -----------------------------
# MAIN LOOP using COT prompt
# -----------------------------
if __name__ == "__main__":
    segment_length = 100
    prediction_length = 5
    step_forward = 100

    mse_errors = []
    
    #stock_list, _ = load_stock_segments(
    #    "data/preprocessed_CAC40.csv",
    #    company="BNP",
    #    start=dt.datetime(2014, 1, 1),
    #    end=dt.datetime(2020, 1, 1)
    #)

    stock_list, _ = load_stock_segments(
    "data/preprocessed_CAC40.csv",
    company="BNP Paribas",
    start=dt.datetime(2014, 1, 1),
    end=dt.datetime(2020, 1, 1)
    )

    
    for i in range(0, len(stock_list) - (segment_length + prediction_length), step_forward):
        print("Index:", i)

        selected = stock_list[i: i + segment_length]
        target = stock_list[i + segment_length: i + segment_length + prediction_length]

        image_path = load_image(selected)

        prompt = generate_prompt_COT_p_gpt(selected, segment_length, prediction_length)
        raw_output = get_response_from_vlm(prompt, image_path)

        print("VLM raw_output:", raw_output)
        print("Ground Truth:", target)
        print("-----------------------------------------------------")



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
    
