from PIL import Image
import json

# Display image in system viewer (for terminal scripts)
def displayImage(image_path):
    img = Image.open(image_path)
    print(f"Opening image: {image_path}")
    print(f"Image size: {img.size}, Mode: {img.mode}")
    img.show()  # Opens in system default image viewer

#load the doc_id_to_image_path.json file
with open("doc_id_to_image_path.json", "r") as f:
    doc_id_to_image_path = json.load(f)



input_doc_id = input("Enter the document id you want to visualize: ")
# display the image for the document with id input_doc_id
displayImage(doc_id_to_image_path[input_doc_id])