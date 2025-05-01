def gcd(a, b):
    """Compute the Greatest Common Divisor of a and b."""
    while b:
        a, b = b, a % b  #Euclidean algorithm
    return a

def mod_inverse(e, phi):
    """Compute the modular multiplicative inverse of e mod phi."""
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a) #
        x = y1 - (b // a) * x1 
        y = x1
        return gcd, x, y
    
    gcd, x, _ = extended_gcd(e, phi)
    if gcd != 1:                                                                 #inverse doesn't exist
        raise ValueError("Modular inverse does not exist")
    return (x % phi + phi) % phi    

def generate_keypair(p, q):
    """Generate RSA public and private key pair given prime numbers p and q."""
    # Compute n = p * q
    n = p * q
    
    # Compute phi (φ) = (p-1) * (q-1)
    phi = (p - 1) * (q - 1)
    
    # Choose e: 1 < e < phi, coprime to phi
    e = 65537  # Common choice for e (must be coprime to phi)
    if gcd(e, phi) != 1:
        raise ValueError("e is not coprime with phi")
    
    # Compute d: modular inverse of e mod phi
    d = mod_inverse(e, phi)
    
    # Return public key (e, n) and private key (d, n)                         ###################
    return (e, n), (d, n)



def encrypt(public_key, plaintext):
    """Encrypt the plaintext using the public key (e, n)."""
    e, n = public_key
    if plaintext >= n:
        raise ValueError("Plaintext too large for modulus n")
    # Ciphertext = plaintext^e mod n
    ciphertext = pow(plaintext, e, n)
    return ciphertext


def decrypt(private_key, ciphertext):
    """Decrypt the ciphertext using the private key (d, n)."""
    d, n = private_key
    # Plaintext = ciphertext^d mod n
    plaintext = pow(ciphertext, d, n)
    return plaintext



# Example usage
if __name__ == "__main__":
    # Choose two small prime numbers (in practice, use large primes, e.g., 2048-bit)
    p = 61
    q = 53
    
    # Generate keys
    public_key, private_key = generate_keypair(p, q)
    e, n = public_key
    d, _ = private_key
    
    print(f"Prime p: {p}")
    print(f"Prime q: {q}")
    print(f"Modulus n: {n}")
    print(f"Public exponent e: {e}")
    print(f"Private exponent d: {d}")
    
    # Example plaintext (must be < n)
    plaintext = 42
    print(f"Original plaintext: {plaintext}")
    
    # Encrypt
    ciphertext = encrypt(public_key, plaintext)
    print(f"Ciphertext: {ciphertext}")
    
    # Decrypt
    decrypted = decrypt(private_key, ciphertext)
    print(f"Decrypted text: {decrypted}")
    
    # Verify
    print(f"Decryption successful: {decrypted == plaintext}")