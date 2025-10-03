import hashlib
import json
from datetime import datetime

# A simplified Block class that includes a proof.
class SimpleBlock:
    def __init__(self, index, timestamp, data, previous_hash, proof):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.proof = proof
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class SimpleBlockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
        self.difficulty = 4 # Number of leading zeros required

    def create_genesis_block(self):
        return SimpleBlock(0, str(datetime.now()), "Genesis Block", "0", 1)

    def get_last_block(self):
        return self.chain[-1]

    def mine_block(self, miner_address):
        last_block = self.get_last_block()
        proof = self.proof_of_work(last_block.hash)
        
        # Reward the miner with a transaction
        self.add_transaction("network", miner_address, 1)
        
        new_block_data = {
            'transactions': self.pending_transactions
        }
        
        new_block = SimpleBlock(
            len(self.chain),
            str(datetime.now()),
            new_block_data,
            last_block.hash,
            proof
        )
        
        self.chain.append(new_block)
        self.pending_transactions = [] # Clear transactions for the next block
        return new_block

    def proof_of_work(self, previous_hash):
        proof = 0
        while True:
            # Create a guess by combining the previous hash and the current proof.
            guess = f'{previous_hash}{proof}'.encode()
            guess_hash = hashlib.sha256(guess).hexdigest()
            if guess_hash[:self.difficulty] == '0' * self.difficulty:
                return proof
            proof += 1

    def add_transaction(self, sender, recipient, amount):
        transaction = {'sender': sender, 'recipient': recipient, 'amount': amount}
        self.pending_transactions.append(transaction)
        return self.get_last_block().index + 1

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]

            if current.previous_hash != previous.hash:
                return False

            if not self.is_valid_proof_for_block(previous.hash, current.proof, current.hash):
                return False
        return True
    
    def is_valid_proof_for_block(self, previous_hash, proof, current_hash):
        guess = f'{previous_hash}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:self.difficulty] == '0' * self.difficulty

# --- Run the demonstration ---
my_blockchain = SimpleBlockchain()

while True:
	# Add some transactions
	my_blockchain.add_transaction("Alice", "Bob", 100)
	my_blockchain.add_transaction("Bob", "Charlie", 50)

	# Mine a new block to include the transactions
	print("Mining new block...")
	newly_mined_block = my_blockchain.mine_block("Miner1")
	print(f"New block mined: {newly_mined_block.index}")

	# Show the chain
	print("\nBlockchain:")
	for block in my_blockchain.chain:
	    print(f"Block #{block.index}:")
	    print(f"  Timestamp: {block.timestamp}")
	    print(f"  Data: {block.data}")
	    print(f"  Previous Hash: {block.previous_hash}")
	    print(f"  Proof: {block.proof}")
	    print(f"  Hash: {block.hash}")
	    print("-" * 20)
	    
	# Validate the chain
	print("\nValidating chain...")
	print(f"Is the blockchain valid? {my_blockchain.is_chain_valid()}")

