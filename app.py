import os
import time
import speech_recognition as sr
import torch

from diffusers import StableDiffusionPipeline


# Create the folder where generated images will be saved
OUTPUT_FOLDER = "generated_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


print("=" * 55)
print("🎤 Speech to Image Generator")
print("Say 'exit' to close the application.")
print("=" * 55)


# Select CPU or NVIDIA GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nUsing device: {device.upper()}")
print("Loading the AI image model...")
print("The first launch may take some time because")
print("the model needs to download.\n")


# Automatically use GPU if available; otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use float16 on GPU and float32 on CPU
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"\nUsing device: {device.upper()}")
print("Loading the AI image model...")
print("The first launch may take some time because")
print("the model needs to download.\n")

# Load the Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
)

# Move the model to the selected device
pipe = pipe.to(device)

# Reduce memory usage
pipe.enable_attention_slicing()


def listen_for_prompt():
    """Listen to the microphone and convert speech into text."""

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        print("\n🎤 Speak your image prompt...")
        print("Listening...")

        # Adjust to background noise
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
            print("\n❌ No speech was detected.")
            return None

    try:
        prompt = recognizer.recognize_google(
            audio
        )

        print(f"\nYou said: {prompt}")

        return prompt

    except sr.UnknownValueError:

        print(
            "\n❌ Sorry, I could not understand "
            "the speech."
        )

        return None

    except sr.RequestError as error:

        print(
            "\n❌ Speech recognition service error:"
        )

        print(error)

        return None


def generate_image(prompt):
    """Generate and save an image from the spoken prompt."""

    print("\n🎨 Generating your image...")
    print("Please wait. This can take a while on CPU.")

    try:

        start_time = time.time()

        image = pipe(
            prompt,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]

        timestamp = int(time.time())

        filename = (
            f"generated_image_{timestamp}.png"
        )

        image_path = os.path.join(
            OUTPUT_FOLDER,
            filename
        )

        image.save(image_path)

        elapsed_time = (
            time.time() - start_time
        )

        print("\n✅ Image generated successfully!")

        print(
            f"📁 Saved to: {image_path}"
        )

        print(
            f"⏱️ Generation time: "
            f"{elapsed_time:.1f} seconds"
        )

        # Open the image automatically
        image.show()

    except Exception as error:

        print(
            "\n❌ Image generation failed:"
        )

        print(error)


def main():

    while True:

        prompt = listen_for_prompt()

        if prompt is None:
            continue

        if prompt.lower() == "exit":

            print(
                "\n👋 Goodbye!"
            )

            break

        generate_image(prompt)


if __name__ == "__main__":
    main()