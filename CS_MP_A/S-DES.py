def permute(bits, table):
    """Apply permutation to bits according to the given table."""
    return [bits[i - 1] for i in table]

def f_k(bits, subkey):
    """Function f_k: Expansion, XOR with subkey, S-boxes, and P4 permutation."""
    # Expansion/permutation
    EP = [4, 1, 2, 3, 2, 3, 4, 1]
    expanded = permute(bits[4:], EP)
    
    # XOR with subkey
    xored = [expanded[i] ^ subkey[i] for i in range(8)]
    
    # S-boxes
    S0 = [
        [1, 0, 3, 2],
        [3, 2, 1, 0],
        [0, 2, 1, 3],
        [3, 1, 3, 2]
    ]
    S1 = [
        [0, 1, 2, 3],
        [2, 0, 1, 3],
        [3, 0, 1, 0],
        [2, 1, 0, 3]
    ]
    
    # Split into two 4-bit parts
    left, right = xored[:4], xored[4:]
    
    # S0: row = (b1b4), col = (b2b3)
    row = (left[0] << 1) + left[3]
    col = (left[1] << 1) + left[2]
    s0_val = S0[row][col]
    
    # S1: row = (b1b4), col = (b2b3)
    row = (right[0] << 1) + right[3]
    col = (right[1] << 1) + right[2]
    s1_val = S1[row][col]
    
    # Convert S-box outputs to binary
    s0_bits = [(s0_val >> 1) & 1, s0_val & 1]
    s1_bits = [(s1_val >> 1) & 1, s1_val & 1]
    
    # P4 permutation
    P4 = [2, 4, 3, 1]
    p4_input = s0_bits + s1_bits
    p4_output = permute(p4_input, P4)
    
    # XOR with left half
    left_half = bits[:4]
    result = [left_half[i] ^ p4_output[i] for i in range(4)]
    
    return result + bits[4:]

def sdes_encrypt(plaintext, k1, k2):
    """Encrypt an 8-bit plaintext using S-DES with two 8-bit subkeys."""
    # Convert plaintext to binary list
    plaintext = [(plaintext >> (7 - i)) & 1 for i in range(8)]
    # Convert subkeys to binary lists
    k1 = [(k1 >> (7 - i)) & 1 for i in range(8)]
    k2 = [(k2 >> (7 - i)) & 1 for i in range(8)]
    
    # Initial permutation (IP)
    IP = [2, 6, 3, 1, 4, 8, 5, 7]
    bits = permute(plaintext, IP)
    
    # First round with k1
    bits = f_k(bits, k1)
    
    # Swap halves
    bits = bits[4:] + bits[:4]
    
    # Second round with k2
    bits = f_k(bits, k2)
    
    # Inverse initial permutation (IP^-1)
    IP_inv = [4, 1, 3, 5, 7, 2, 8, 6]
    ciphertext = permute(bits, IP_inv)
    
    # Convert to integer
    result = 0
    for bit in ciphertext:
        result = (result << 1) | bit
    return result

def sdes_decrypt(ciphertext, k1, k2):
    """Decrypt an 8-bit ciphertext using S-DES with two 8-bit subkeys."""
    # Convert ciphertext to binary list
    ciphertext = [(ciphertext >> (7 - i)) & 1 for i in range(8)]
    # Convert subkeys to binary lists
    k1 = [(k1 >> (7 - i)) & 1 for i in range(8)]
    k2 = [(k2 >> (7 - i)) & 1 for i in range(8)]
    
    # Initial permutation (IP)
    IP = [2, 6, 3, 1, 4, 8, 5, 7]
    bits = permute(ciphertext, IP)
    
    # First round with k2
    bits = f_k(bits, k2)
    
    # Swap halves
    bits = bits[4:] + bits[:4]
    
    # Second round with k1
    bits = f_k(bits, k1)
    
    # Inverse initial permutation (IP^-1)
    IP_inv = [4, 1, 3, 5, 7, 2, 8, 6]
    plaintext = permute(bits, IP_inv)
    
    # Convert to integer
    result = 0
    for bit in plaintext:
        result = (result << 1) | bit
    return result

# Test with provided values
if __name__ == "__main__":
    # Provided values
    k1 = 0b10100100  # Key-1: 1 0 1 0 0 1 0 0
    k2 = 0b01000011  # Key-2: 0 1 0 0 0 0 1 1
    plaintext = 0b10010111  # Plaintext: 1 0 0 1 0 1 1 1
    expected_ciphertext = 0b00111000  # Ciphertext: 0 0 1 1 1 0 0 0
    expected_decrypted = 0b10010111  # Decrypted: 1 0 0 1 0 1 1 1
    
    # Encrypt
    ciphertext = sdes_encrypt(plaintext, k1, k2)
    print(f"Plaintext: {format(plaintext, '08b')}")
    print(f"Key-1: {format(k1, '08b')}")
    print(f"Key-2: {format(k2, '08b')}")
    print(f"Ciphertext: {format(ciphertext, '08b')}")
    
    # # Verify encryption
    # if ciphertext == expected_ciphertext:
    #     print("Encryption successful: Ciphertext matches expected.")
    # else:
    #     print("Encryption failed: Ciphertext does not match expected.")
    
    # # Decrypt
    # decrypted = sdes_decrypt(ciphertext, k1, k2)
    # print(f"Decrypted Text: {format(decrypted, '08b')}")
    
    # # Verify decryption
    # if decrypted == expected_decrypted:
    #     print("Decryption successful: Decrypted text matches plaintext.")
    # else:
    #     print("Decryption failed: Decrypted text does not match plaintext.")