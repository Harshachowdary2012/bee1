def hex2bin(s):
    return bin(int(s, 16))[2:].zfill(len(s)*4)

def bin2hex(s):
    return hex(int(s, 2))[2:].upper().zfill(len(s)//4)

def xor(a, b):
    return ''.join('0' if x == y else '1' for x, y in zip(a, b))

# Simple fixed S-box (4-bit substitution for demonstration)
sbox = {
    '0000': '1110', '0001': '0100', '0010': '1101', '0011': '0001',
    '0100': '0010', '0101': '1111', '0110': '1011', '0111': '1000',
    '1000': '0011', '1001': '1010', '1010': '0110', '1011': '1100',
    '1100': '0101', '1101': '1001', '1110': '0000', '1111': '0111'
}

# Inverse S-box for decryption
inv_sbox = {v: k for k, v in sbox.items()}

def substitute(block, box):
    result = ''
    for i in range(0, len(block), 4):
        nibble = block[i:i+4]
        result += box[nibble]
    return result

def shift_rows(block):
    # Simple row shift: swap halves (8 bits each)
    return block[8:] + block[:8]

def inv_shift_rows(block):
    return block[8:] + block[:8]

def add_round_key(block, round_key):
    return xor(block, round_key)

def simple_aes_encrypt(pt_hex, round_keys):
    block = hex2bin(pt_hex)
    print(f"Initial Block: {block}")

    for i, rk in enumerate(round_keys):
        block = substitute(block, sbox)
        block = shift_rows(block)
        block = add_round_key(block, rk)
        print(f"Round {i+1}: {bin2hex(block)}")

    return bin2hex(block)

def simple_aes_decrypt(ct_hex, round_keys):
    block = hex2bin(ct_hex)
    print(f"Initial Cipher Block: {block}")

    for i, rk in enumerate(reversed(round_keys)):
        block = add_round_key(block, rk)
        block = inv_shift_rows(block)
        block = substitute(block, inv_sbox)
        print(f"Round {i+1}: {bin2hex(block)}")

    return bin2hex(block)

# Example fixed 16-bit round keys (simple example)
round_keys = [
    '0011001100110011',
    '1100110011001100',
    '1010101010101010',
    '0101010101010101'
]

# Input and process
pt = input("Enter plaintext (4 hex chars): ").upper()  # 16 bits = 4 hex chars
cipher_text = simple_aes_encrypt(pt, round_keys)
print("Ciphertext (hex):", cipher_text)

decrypted_text = simple_aes_decrypt(cipher_text, round_keys)
print("Decrypted Plaintext (hex):", decrypted_text)
