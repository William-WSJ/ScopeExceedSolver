import json
import os
from openai import OpenAI


# Enter your OpenAI API Key here
api_key_gpt = ""
client_gpt = OpenAI(api_key = api_key_gpt)

def upload_file_2_openai(file_path):
    client = client_gpt

    batch_input_file = client.files.create(
        file = open(file_path, "rb"),
        purpose = "batch"
    )

    print(batch_input_file)
    return batch_input_file


def batch_test(file_path):
    batch_input_file = upload_file_2_openai(file_path)
    batch_input_file_id = batch_input_file.id
    client = client_gpt
    batch_object = client.batches.create(
        input_file_id = batch_input_file_id,
        endpoint = "/v1/chat/completions",
        completion_window = "24h",
        metadata = {
            "description": "test job"
        }
    )
    print(batch_object)
    if batch_object is not None:
        print(
            f"Batch job submitted successfully. Metadata: {batch_object.metadata}. "
            f"Please save the batch job ID: {batch_object.id}"
        )
    else:
        print("Batch job submission failed. Please try again.")
    return batch_object

def check_batch_job_status(batch_id):
    client = client_gpt
    batch_object = client.batches.retrieve(batch_id)
    print(batch_object)
    if batch_object is not None:
        if batch_object.status != "completed":
            print(
                f"Batch job ID: {batch_id}, current status: {batch_object.status}, "
                f"progress: {batch_object.request_counts.completed}/{batch_object.request_counts.total}, "
                f"failed requests: {batch_object.request_counts.failed}. Please wait..."
            )
        else:
            print(f"Batch job ID: {batch_id} completed. Output file ID: {batch_object.output_file_id}")
            return batch_object.files[ 0 ].id
    else:
        print("Invalid batch ID. No batch job was found.")
        return -1

def get_file_from_openai(file_id):
    client = client_gpt
    file_response = client.files.content(file_id)
    text = file_response.text
    print(text)
