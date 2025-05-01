def mod_inverse(a, p):
    """Compute the modular multiplicative inverse of a modulo p."""
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    gcd, x, _ = extended_gcd(a, p)
    if gcd != 1:
        raise ValueError("Modular inverse does not exist")
    return (x % p + p) % p

def point_addition(P, Q, p, a):
    """Add two points P and Q on the elliptic curve y^2 = x^3 + ax + b mod p."""
    if P is None:  # P + ∞ = Q
        return Q
    if Q is None:  # Q + ∞ = P
        return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2 and y1 == -y2 % p:  # P + (-P) = ∞
        return None
    
    if P == Q:  # Point doubling
        # Slope = (3x1^2 + a) / (2y1)
        num = (3 * x1 * x1 + a) % p
        denom = (2 * y1) % p
        m = (num * mod_inverse(denom, p)) % p
    else:  # Point addition
        # Slope = (y2 - y1) / (x2 - x1)
        num = (y2 - y1) % p
        denom = (x2 - x1) % p
        m = (num * mod_inverse(denom, p)) % p
    
    # x3 = m^2 - x1 - x2
    x3 = (m * m - x1 - x2) % p
    # y3 = m(x1 - x3) - y1
    y3 = (m * (x1 - x3) - y1) % p
    
    return (x3, y3)

def scalar_multiply(k, P, p, a):
    """Multiply point P by scalar k on the elliptic curve."""
    if k == 0 or P is None:
        return None
    result = None
    temp = P
    while k > 0:
        if k & 1:  # If k is odd, add temp to result
            result = point_addition(result, temp, p, a)
        temp = point_addition(temp, temp, p, a)  # Double temp
        k >>= 1
    return result

def ecc_diffie_hellman(p, a, b, G, dA, dB):
    """Perform Elliptic Curve Diffie-Hellman key exchange.
    
    Args:
        p: Prime modulus
        a, b: Curve coefficients (y^2 = x^3 + ax + b)
        G: Generator point (x, y)
        dA: Alice's private key
        dB: Bob's private key
    
    Returns:
        Tuple of (Alice's public key, Bob's public key, shared secret)
    """
    # Alice computes her public key: QA = dA * G
    QA = scalar_multiply(dA, G, p, a)
    
    # Bob computes his public key: QB = dB * G
    QB = scalar_multiply(dB, G, p, a)
    
    # Alice computes shared secret: dA * QB
    alice_secret = scalar_multiply(dA, QB, p, a)
    
    # Bob computes shared secret: dB * QA
    bob_secret = scalar_multiply(dB, QA, p, a)
    
    return QA, QB, alice_secret, bob_secret

# Example usage
if __name__ == "__main__":
    # Curve parameters: y^2 = x^3 + 2x + 2 mod 17
    p = 17  # Prime modulus
    a = 2   # Coefficient a
    b = 2   # Coefficient b
    G = (5, 1)  # Generator point (must be on curve)
    
    # Private keys (chosen for demonstration)
    dA = 6  # Alice's private key
    dB = 9  # Bob's private key
    
    # Perform ECDH
    QA, QB, alice_secret, bob_secret = ecc_diffie_hellman(p, a, b, G, dA, dB)
    
    # Print results
    print(f"Curve: y^2 = x^3 + {a}x + {b} mod {p}")
    print(f"Generator point G: {G}")
    print(f"Alice's private key: {dA}")
    print(f"Alice's public key QA: {QA}")
    print(f"Bob's private key: {dB}")
    print(f"Bob's public key QB: {QB}")
    print(f"Alice's computed shared secret: {alice_secret}")
    print(f"Bob's computed shared secret: {bob_secret}")
    print(f"Shared secret matches: {alice_secret == bob_secret}")