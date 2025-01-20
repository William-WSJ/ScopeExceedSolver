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
            f"本次任务元数据:{batch_object.metadata},批量任务提交成功!请保存牢记此次批量任务id:{batch_object.id}")
    else:
        print("批量任务提交失败!请重试")
    return batch_object

def check_batch_job_status(batch_id):
    client = client_gpt
    batch_object = client.batches.retrieve(batch_id)
    print(batch_object)
    if batch_object is not None:
        if batch_object.status != "completed":
            print(
                f"批量任务id:{batch_id},目前状态:{batch_object.status},任务进度:{batch_object.request_counts.completed}/{batch_object.request_counts.total},任务失败个数:{batch_object.request_counts.failed},请继续等待...")
        else:
            print(f"批量任务id:{batch_id}, 已完成,输出文件id为:{batch_object.output_file_id}")
            return batch_object.files[ 0 ].id
    else:
        print("batch_id输入错误,批量任务为空")
        return -1

def get_file_from_openai(file_id):
    client = client_gpt
    file_response = client.files.content(file_id)
    text = file_response.text
    print(text)