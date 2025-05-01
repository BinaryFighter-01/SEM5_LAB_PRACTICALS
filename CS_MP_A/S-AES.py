# S-AES S-Box and Inverse S-Box (for reference, only S-Box used here)
SBOX = [9, 4, 10, 11, 13, 1, 8, 5, 6, 2, 0, 3, 12, 14, 15, 7]

# Helper function: XOR two binary strings of equal length
def xor_bits(a, b):
    return format(int(a, 2) ^ int(b, 2), f'0{len(a)}b')

# Rotate nibble (swap two 4-bit parts of an 8-bit string)
def rot_nib(bits):
    return bits[4:] + bits[:4]

# Substitute nibbles using S-Box
def sub_nib(bits):
    nib1 = SBOX[int(bits[:4], 2)]   # First 4 bits
    nib2 = SBOX[int(bits[4:], 2)]   # Last 4 bits
    return format(nib1, '04b') + format(nib2, '04b')\

# Key expansion: Generate round keys from 16-bit key
def expand_key(key):
    w0 = key[:8]    # First 8 bits
    w1 = key[8:]    # Last 8 bits
    rcon = "10000000"  # Round constant
    w2 = xor_bits(w0, xor_bits(sub_nib(rot_nib(w1)), rcon))
    return w0 + w1, w2 + w0, w1 + w2  # K0, K1, K2

# AddRoundKey: XOR state with round key
def add_round_key(state, round_key):
    return xor_bits(state, round_key)

# SubNib: Apply S-Box to each 4-bit nibble of 16-bit state
def sub_nib_full(state):
    s00 = SBOX[int(state[0:4], 2)]
    s01 = SBOX[int(state[4:8], 2)]
    s10 = SBOX[int(state[8:12], 2)]
    s11 = SBOX[int(state[12:16], 2)]
    return format(s00, '04b') + format(s01, '04b') + format(s10, '04b') + format(s11, '04b')

# ShiftRows: Swap s01 and s11
def shift_rows(state):
    s00 = state[0:4]
    s01 = state[4:8]
    s10 = state[8:12]
    s11 = state[12:16]
    return s00 + s11 + s10 + s01  # s01 and s11 swapped

# MixColumns: Matrix multiplication in GF(2^4)
def gf_mult(a, b):
    # Multiplication in GF(2^4) with modulo x^4 + x + 1
    p = 0
    for _ in range(4):
        if b & 1:
            p ^= a
        carry = a & 8  # Check if overflow
        a <<= 1
        if carry:
            a ^= 0b10011  # x^4 + x + 1
        a &= 0b1111    # Keep 4 bits
        b >>= 1
    return p

def mix_columns(state):
    s0, s1 = int(state[0:4], 2), int(state[4:8], 2)    # First row
    s2, s3 = int(state[8:12], 2), int(state[12:16], 2) # Second row
    # [1 4] [s0 s1] = [s0+4s1, s1+4s0]
    # [4 1] [s2 s3]   [s2+4s3, s3+4s2]
    s0_new = s0 ^ gf_mult(4, s1)
    s1_new = s1 ^ gf_mult(4, s0)
    s2_new = s2 ^ gf_mult(4, s3)
    s3_new = s3 ^ gf_mult(4, s2)
    return format(s0_new, '04b') + format(s1_new, '04b') + format(s2_new, '04b') + format(s3_new, '04b')

# S-AES encryption function
def s_aes_encrypt(plaintext, key):
    # Validate inputs
    if len(plaintext) != 16 or len(key) != 16:
        raise ValueError("Plaintext and key must be 16-bit binary strings")
    
    # Generate round keys
    K0, K1, K2 = expand_key(key)
    
    # Round 0: AddRoundKey
    state = add_round_key(plaintext, K0)
    
    # Round 1
    state = sub_nib_full(state)
    state = shift_rows(state)
    state = mix_columns(state)
    state = add_round_key(state, K1)
    
    # Round 2 (no MixColumns)
    state = sub_nib_full(state)
    state = shift_rows(state)
    state = add_round_key(state, K2)
    
    return state

# Main function with example inputs
def main():
    # Example inputs (you can replace these)
    plaintext = "0110100101100011"  # 16-bit plaintext
    key = "0100101011110101"      # 16-bit key
    
    # Encrypt the plaintext
    ciphertext = s_aes_encrypt(plaintext, key)
    
    # Print results (only plaintext and ciphertext)
    print(f"Plaintext:  {plaintext}")
    print(f"Ciphertext: {ciphertext}")

# Run the program
if __name__ == "__main__":
    main()