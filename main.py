import hashlib
import json
from time import time
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

class Blockchain:
    def __init__(self, storage_file='blockchain.json'):
        self.chain = []
        self.current_transactions = []
        self.storage_file = storage_file
        self.load_chain()
        if not self.chain:  # Create genesis block if chain is empty
            self.new_block(previous_hash='1', proof=100)

    def new_block(self, proof, previous_hash=None):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }
        self.current_transactions = []
        self.chain.append(block)
        self.save_chain()
        return block

    def new_transaction(self, image_hash, verification_result):
        self.current_transactions.append({
            'image_hash': image_hash,
            'verification_result': verification_result,
        })

    def hash(self, block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def last_block(self):
        return self.chain[-1]

    def save_chain(self):
        with open(self.storage_file, 'w') as f:
            json.dump(self.chain, f, indent=4)

    def load_chain(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r') as f:
                self.chain = json.load(f)

    def display_chain(self):
        for block in self.chain:
            print(f"Block {block['index']}:")
            print(f"Timestamp: {block['timestamp']}")
            print(f"Transactions: {block['transactions']}")
            print(f"Proof: {block['proof']}")
            print(f"Previous Hash: {block['previous_hash']}")
            print("\n")

# Load pre-trained deep fake detection model
from keras.models import model_from_json

# Load model architecture
with open('model_architecture.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Load weights
model.load_weights(r'D:\Code\TY\BlockChain\celebs_vit_weights.h5')


def is_image_fake(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Adjust size to model's input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    # Assuming the model returns a probability where > 0.5 is fake
    return 'Fake' if prediction[0] > 0.5 else 'Real'

# Main flow
blockchain = Blockchain()

def detect_and_store(image_path):
    image_hash = hashlib.sha256(open(image_path, 'rb').read()).hexdigest()
    result = is_image_fake(image_path)
    blockchain.new_transaction(image_hash, result)
    blockchain.new_block(proof=100)  # Simplified proof

# Example usage
image_path = 'images/image3.jpeg'
detect_and_store(image_path)

# Display the blockchain details
blockchain.display_chain()
