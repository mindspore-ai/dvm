#!/usr/bin/env python3
import sys
import struct
import subprocess

def find_section_offset(binary_path, section_name):
    result = subprocess.run(
        ['objdump', '-h', binary_path],
        capture_output=True, text=True
    )
    for line in result.stdout.split('\n'):
        if section_name in line and 'Idx' not in line:
            parts = line.split()
            for part in reversed(parts):
                if len(part) >= 8 and all(c in '0123456789abcdef' for c in part):
                    try:
                        return int(part, 16)
                    except ValueError:
                        continue
    return None

def find_aiv_type_value_offset(binary_path):
    section_name = '.ascend.meta.dvm_mix_aiv'
    section_offset = find_section_offset(binary_path, section_name)
    with open(binary_path, 'rb') as f:
        f.seek(section_offset)
        section_size = 48
        pos = 0
        while pos < section_size:
            type_val = struct.unpack('<H', f.read(2))[0]
            reserved = struct.unpack('<H', f.read(2))[0]
            value = struct.unpack('<I', f.read(4))[0]
            if type_val == 12:
                value_offset = section_offset + pos + 4
                return value_offset
            pos += 8
    raise ValueError("aiv_type meta not found")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <binary_file>")
        sys.exit(1)
    binary_path = sys.argv[1]
    offset = find_aiv_type_value_offset(binary_path)
    print(f"extern const unsigned int g_meta_aiv_type_value_offset_c310 = {offset};")