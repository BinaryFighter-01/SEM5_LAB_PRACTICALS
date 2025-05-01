# Diffie-Hellman Key Exchange Implementation

# Modular exponentiation function (efficiently computes base^exp mod modulus)
def mod_exp(base, exp, modulus):
    result = 1
    base = base % modulus
    while exp > 0:
        if exp & 1:  # If exponent is odd
            result = (result * base) % modulus
        base = (base * base) % modulus
        exp >>= 1
    return result

# Diffie-Hellman key exchange function
def diffie_hellman(p, g, a, b):
    # Alice computes her public key: A = g^a mod p
    A = mod_exp(g, a, p)
    
    # Bob computes his public key: B = g^b mod p
    B = mod_exp(g, b, p)
    
    # Alice computes shared secret: S = B^a mod p
    shared_secret_alice = mod_exp(B, a, p)
    
    # Bob computes shared secret: S = A^b mod p
    shared_secret_bob = mod_exp(A, b, p)
    
    # Verify they match (should always be true)
    assert shared_secret_alice == shared_secret_bob, "Shared secrets do not match!" 
    
    return A, B, shared_secret_alice

# Main function with example inputs   
def main():
    # Public parameters (small for demonstration; use large prime in practice)
    p = 23  # Prime modulus
    g = 5   # Generator
    
    # Private keys (chosen secretly by Alice and Bob)
    a = 6   # Alice's private key
    b = 15  # Bob's private key
    
    # Perform Diffie-Hellman key exchange
    A, B, shared_secret = diffie_hellman(p, g, a, b) 
    
    # Print results
    print(f"Prime (p):          {p}")
    print(f"Generator (g):      {g}")
    print(f"Alice's Public Key (A): {A}")
    print(f"Bob's Public Key (B):   {B}")
    print(f"Shared Secret (S):      {shared_secret}") 

# Run the program
if __name__ == "__main__":
    main()
    