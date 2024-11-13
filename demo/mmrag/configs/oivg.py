import json
import boto3
import botocore
from PIL import Image
import os
import base64
import io
from random import randint
#from datetime import datetime
import datetime
from IPython.display import display
from dateutil.tz import tzlocal
import time

def assume_role(account_id, role_name, session_name):
    """
    Assume an IAM role in another account

    Parameters:
    account_id (str): The AWS account ID where the role exists
    role_name (str): The name of the role to assume
    session_name (str): An identifier for the assumed role session

    Returns:
    boto3.Session: A boto3 session with the assumed role credentials
    """
    # Create an STS client
    sts_client = boto3.client('sts')

    # Construct the role ARN
    role_arn = f'arn:aws:iam::{account_id}:role/{role_name}'

    try:
        # Assume the role
        assumed_role_object = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName=session_name
        )

        # Extract the temporary credentials
        credentials = assumed_role_object['Credentials']

        # Create a new session with the temporary credentials
        session = boto3.Session(
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken']
        )

        return session

    except Exception as e:
        print(f"Error assuming role: {str(e)}")
        raise

def assumed_role_session(role_arn: str, base_session: botocore.session.Session = None):
    base_session = base_session or boto3.session.Session()._session
    fetcher = botocore.credentials.AssumeRoleCredentialFetcher(
        client_creator = base_session.create_client,
        source_credentials = base_session.get_credentials(),
        role_arn = role_arn,
        extra_args = {
        #    'RoleSessionName': None # set this if you want something non-default
        }
    )
    creds = botocore.credentials.DeferredRefreshableCredentials(
        method = 'assume-role',
        refresh_using = fetcher.fetch_credentials,
        time_fetcher = lambda: datetime.datetime.now(tzlocal())
    )
    botocore_session = botocore.session.Session()
    botocore_session._credentials = creds
    return boto3.Session(botocore_session = botocore_session)


def t2i_olympus(prompt:str, neg_prompt:str, cfgScale:float=7.0, num_image: int=1, width:int=1280, height:int=768, quality: str='premium'  ):
    # Create a new BedrockRuntime client.
    bedrock_runtime = assumed_role_session("arn:aws:iam::905418197933:role/ovg_developer").client(
        "bedrock-runtime",
        region_name="us-east-1",  # You must use us-east-1 during the beta period.
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    )

    # Configure the inference parameters.
    inference_params = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt,
            "negativeText": neg_prompt  # A list of items that should not appear in the image.
        },
        "imageGenerationConfig": {
            "numberOfImages": num_image,  # Number of images to generate, up to 5.
            "width": width,  # See README for supported resolutions.
            "height": height,  # See README for supported resolutions.
            "cfgScale": cfgScale,  # How closely the prompt will be followed.
            "quality": quality,  # "standard" or "premium"
            "seed": randint(
                0, 2147483646
            ),  # Using a random seed value guarantees we get different results each time this code is executed.
        },
    }
    
    # Display the random seed.
    print(f"Generating with seed: {inference_params['imageGenerationConfig']['seed']}")
    
    start_time = datetime.datetime.now()
    
    # Invoke the model.
    response = bedrock_runtime.invoke_model(
        modelId="amazon.olympus-image-generator-v1:0",
        body=json.dumps(inference_params),
    )
    # Convert the JSON-formatted response to a dictionary.
    response_body = json.loads(response["body"].read())
    images = response_body["images"]
    return_images = []
    for i in range(len(images)):
        image_data = images[i]
        image_bytes = base64.b64decode(image_data)
        #image = Image.open(io.BytesIO(image_bytes))
        return_images.append(Image.open(io.BytesIO(image_bytes)))

    return return_images

def download_video_for_invocation_arn(invocation_arn: str, role_arn:str, bucket_name: str="ovg-videos", destination_folder:str="./output"):
    """
    This function downloads the video file for the given invocation ARN.
    """
    invocation_id = invocation_arn.split("/")[-1]

    # Create the local file path
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{timestamp}_{invocation_id}.mp4"
    import os

    output_folder = os.path.abspath(destination_folder)
    local_file_path = os.path.join(output_folder, file_name)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Create an STS client
    sts_client = boto3.client('sts')

    # Assume the specified role
    assumed_role = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName='CallBedrockOVG'
    )
    # Extract temporary credentials
    credentials = assumed_role['Credentials']
    
    # Create an S3 client
    s3 = boto3.client("s3",
                        aws_access_key_id=credentials['AccessKeyId'],
                        aws_secret_access_key=credentials['SecretAccessKey'],
                        aws_session_token=credentials['SessionToken'],
                     )

    # List objects in the specified folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=invocation_id)
    
    # Find the first MP4 file and download it.
    for obj in response.get("Contents", []):
        object_key = obj["Key"]
        if object_key.endswith(".mp4"):
            print(f"""Downloading "{object_key}"...""")
            s3.download_file(bucket_name, object_key, local_file_path)
            print(f"Downloaded to {local_file_path}")
            return local_file_path

    # If we reach this point, no MP4 file was found.
    print(f"Problem: No MP4 file was found in S3 at {bucket_name}/{invocation_id}")

def t2v_ovg(video_prompt:str, role_arn:str, v_length:int=6, s3_destination_bucket:str="ovg-videos", region='us-east-1'):
    """
    Assume an IAM role and create an S3 bucket

    :param role_arn: ARN of the IAM role to assume
    :param bucket_name: Name of the S3 bucket to create
    :param region: AWS region to create the bucket in
    """
    # Configure the inference parameters.
    model_input = {
        "taskType": "TEXT_VIDEO",  # This is the only task type supported in Beta. Other tasks types will be supported at launch
        "textToVideoParams": {"text": video_prompt},
        "videoGenerationConfig": {
            "seconds": v_length,  # 6 is the only supported value currently.
            "fps": 24,  # 24 is the only supported value currently.
            "dimension": "1280x720",  # "1280x720" is the only supported value currently.
            "seed": randint(
                -2147483648, 2147483648
            ),  # A random seed guarantees we'll get a different result each time this code runs.
        },
    }
    
    # Create an STS client
    sts_client = boto3.client('sts')

    # Assume the specified role
    assumed_role = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName='CallBedrockOVG'
    )

    # Extract temporary credentials
    credentials = assumed_role['Credentials']

    # Call Bedrock
    bedrock_runtime = boto3.client(
        "bedrock-runtime",
        region_name=region,  # You must use us-east-1 during the beta period.
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
    )
    invocation_jobs = bedrock_runtime.start_async_invoke_model(
        modelId="amazon.olympus-video-generator-v1:0",
        modelInput=model_input,
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{s3_destination_bucket}"}},
    )
    

    # Check the status of the job until it's complete.
    invocation_arn = invocation_jobs["invocationArn"]
    file_path = ''
    while True:
        invocation_job = bedrock_runtime.get_async_invoke_model(
            invocationArn=invocation_arn
        )
    
        status = invocation_job["status"]
        if status == "InProgress":
            time.sleep(1)
        elif status == "Completed":
            print("\nJob complete!")
            # Save the video to disk.
            s3_bucket = (
                invocation_job["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
                .split("//")[1]
                .split("/")[0]
            )
            file_path = download_video_for_invocation_arn(invocation_arn, role_arn, s3_destination_bucket, "./output")
            break
        else:
            print("\nJob failed!")
            print("\nResponse:")
            print(json.dumps(invocation_job, indent=2, default=str))
    return file_path

if __name__ == "__main__":
    prompt = "A high res, 4k image of fall foliage with tree on different layers ofcolors from red, orange, yellow to all season green with snow covered mountain peaks in the backgroung and running creak at the front vivid color and photoreastic"
    neg_prompt = "human, single color tree"
    images = t2i_olympus(prompt, neg_prompt, num_image=3)
    print(f"Image size: {len(images)}")
    video_prompt = "Closeup of a scence of fall foliage in Sierra with snow covered mountain peaks and running stream, frothy gentle wind blowing tree leaves. Camera zoom in."
    file_name = t2v_ovg(video_prompt=video_prompt, role_arn="arn:aws:iam::905418197933:role/ovg_developer")
    print(f"Video file: {file_name}")