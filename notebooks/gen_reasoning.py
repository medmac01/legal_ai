import pandas as pd
from textwrap import dedent
from ollama import Client

new_df = pd.read_csv("templates_salmane/tenancy_processed.csv")
new_df


def get_llm_response(llm_pipeline: Client, prompt: str, model: str = None):
    """Gets a response from the LLM using the specified model."""
    if model is None:
        model = "deepseek-r1:32b"
        if model is None:
            raise ValueError("No model specified and OLLAMA_MODEL environment variable is not set")
    
    # print(f"Generating LLM response with model: {model}")
    sequences = llm_pipeline.generate(
        model=model,
        prompt=prompt,
    )
    return sequences['response']


llm_pipeline = Client(host="http://localhost:11434")


import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# Create two LLM clients for different GPUs
llm_pipeline_gpu0 = Client(host="http://localhost:11434")
llm_pipeline_gpu1 = Client(host="http://localhost:11434")  # Assuming second Ollama instance on different port

# Or if using single Ollama instance, we'll use threading to manage requests
llm_pipelines = [llm_pipeline_gpu0, llm_pipeline_gpu1]

def process_row_batch(batch_data):
    """Process a batch of rows with a specific GPU pipeline"""
    pipeline_idx, rows_batch = batch_data
    pipeline = llm_pipelines[pipeline_idx % len(llm_pipelines)]
    
    results = []
    for i, row in rows_batch:
        system = row['system']
        user_prompt = row['user_prompt']
        context = row['context']
        
        generate_reasoning = f"""
        {system}\n
        {context}\n
        {user_prompt}

        """

        response = get_llm_response(pipeline, generate_reasoning, model="qwen3:8b")
        reasoning_trace = response.split("<think>")[1].split("</think>")[0] if "<think>" in response else "No reasoning trace generated."
        
        results.append((i, reasoning_trace))
        print(f"GPU {pipeline_idx}: Generated reasoning trace for row {i}")
    
    return results

# Split the dataframe into batches for each GPU
batch_size = len(new_df) // 4
batches = [
    (0, list(new_df.iloc[:batch_size].iterrows())),
    (1, list(new_df.iloc[batch_size:2*batch_size].iterrows())),
    (2, list(new_df.iloc[2*batch_size:3*batch_size].iterrows())),
    (3, list(new_df.iloc[3*batch_size:].iterrows()))
]

# Process batches in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_row_batch, batch) for batch in batches]
    
    # Collect results
    for future in futures:
        results = future.result()
        for i, reasoning_trace in results:
            new_df.at[i, 'reasoning_trace'] = reasoning_trace


# Save the updated DataFrame with reasoning traces
new_df.to_csv("templates_salmane/tenancy_processed_with_reasoning.csv", index=False)
print("Reasoning traces saved to 'tenancy_processed_with_reasoning.csv'.")