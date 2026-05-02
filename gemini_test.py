from google import genai
from google.genai import types
from PIL import Image

client = genai.Client(
    vertexai=True, 
    project="project-baba0887-0dd6-4603-91b", # Using the project from your terminal output
)

input_image = Image.open("/projects/bfuj/lzhang49/multimodal-conv-bench/data/coco/images/000000000632.jpg")
prompt = ("Create a picture of adding a banana in this image")
response = client.models.generate_content(
    model="gemini-3.1-flash-image-preview",
    contents=[prompt, input_image],
)

for part in response.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        image = part.as_image()
        image.save("generated_image.png")