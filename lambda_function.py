import boto3
import json

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2"
)

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        prompt = body.get('prompt', 'Hello! How can I help you today?')
        temperature = body.get('temperature', 0.7)  # Default temperature
        max_tokens = body.get('max_tokens', 200)    # Default token count
        
        # Prepare the request body for Messages API (required for Claude Sonnet 4)
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        # Invoke the Bedrock model using Messages API format
        response = bedrock_runtime.invoke_model(
            body=json.dumps(request_body),
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",  # Corrected model ID
            contentType="application/json"
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        
        # Extract the generated text from the response
        generated_text = response_body['content'][0]['text']
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'response': generated_text,
                'usage': response_body.get('usage', {}),
                'model': response_body.get('model', '')
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f'An error occurred: {str(e)}'
            })
        }
