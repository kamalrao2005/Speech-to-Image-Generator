import os
import time
import torch
import speech_recognition as sr
from diffusers import StableDiffusionPipeline

# Configuration

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_FOLDER = "generated_images"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Banner


print("=" * 60)
print("🎤 Speech-to-Image Generator")
print("=" * 60)
print("Say 'exit' anytime to close the application.\n")


# Device Selection

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"🖥️  Device : {device.upper()}")
print(f"🧠 Model  : Stable Diffusion v1.5")
print("\nLoading AI model...")
print("The first launch may take several minutes because")
print("the model will be downloaded from Hugging Face.\n")

# Load Stable Diffusion Model

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype
)

pipe = pipe.to(device)

# Reduce memory usage
pipe.enable_attention_slicing()

print("✅ Model loaded successfully!\n")

# Speech Recognition


recognizer = sr.Recognizer()


def listen_for_prompt():
    """
    Listen to the user's voice and convert it to text.
    Returns:
        str | None
    """

    with sr.Microphone() as source:

        print("\n🎤 Speak your image prompt...")
        print("Listening...")

        recognizer.adjust_for_ambient_noise(
            source,
            duration=1
        )

        try:
            audio = recognizer.listen(
                source,
                timeout=15,
                phrase_time_limit=20
            )

        except sr.WaitTimeoutError:
            print("\n❌ No speech detected.")
            return None

    try:
        prompt = recognizer.recognize_google(audio)

        print(f"\n🗣️ You said: {prompt}")

        return prompt

    except sr.UnknownValueError:
        print("\n❌ Could not understand your speech.")
        return None

    except sr.RequestError as e:
        print("\n❌ Speech Recognition Error")
        print(e)
        return None

# Image Generation

def generate_image(prompt):
    """
    Generate an AI image from a text prompt.
    """

    print("\n🎨 Generating image...")
    print("Please wait...\n")

    start_time = time.time()

    try:

        image = pipe(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]

        timestamp = int(time.time())

        filename = f"generated_image_{timestamp}.png"

        image_path = os.path.join(
            OUTPUT_FOLDER,
            filename
        )

        image.save(image_path)

        generation_time = time.time() - start_time

        print("✅ Image generated successfully!")
        print(f"📁 Saved to : {image_path}")
        print(f"⏱️ Generation Time : {generation_time:.2f} seconds")

        # Open image automatically
        image.show()

    except Exception as e:

        print("\n❌ Failed to generate image.")
        print(e)


# Main Program

def main():

    while True:

        prompt = listen_for_prompt()

        if prompt is None:
            continue

        if prompt.lower() == "exit":

            print("\n👋 Thank you for using Speech-to-Image Generator!")
            break

        generate_image(prompt)


# Run Application

if __name__ == "__main__":
    main()