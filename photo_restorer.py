from dotenv import load_dotenv
import replicate

load_dotenv()


def predict_image(filename):
    output = replicate.run(
        "microsoft/bringing-old-photos-back-to-life:c75db81db6cbd809d93cc3b7e7a088a351a3349c9fa02b6d393e35e0d51ba799",
        input={
            "HR": True,
            "image": open(filename, "rb"),
            "with_scratch": True
        }
    )
    print(output)
    return output
