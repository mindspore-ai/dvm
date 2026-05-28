#!/usr/bin/env python3
import sys
import struct
import subprocess

def find_section_info(binary_path, section_name):
    result = subprocess.run(
        ['objdump', '-h', binary_path],
        capture_output=True, text=True, check=True
    )

    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[1] == section_name:
            size = int(parts[2], 16)
            file_off = int(parts[5], 16)
            return file_off, size

    raise ValueError(f"section {section_name} not found")

def find_aiv_type_value_offset(binary_path):
    section_name = '.ascend.meta.dvm_mix_aiv'
    section_offset, section_size = find_section_info(binary_path, section_name)

    with open(binary_path, 'rb') as f:
        f.seek(section_offset)
        pos = 0
        while pos + 8 <= section_size:
            data = f.read(8)
            type_val, reserved, value = struct.unpack('<HHI', data)
            if type_val == 12:
                return section_offset + pos + 4
            pos += 8

    raise ValueError("aiv_type meta not found")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <binary_file>")
        sys.exit(1)
    binary_path = sys.argv[1]
    offset = find_aiv_type_value_offset(binary_path)
    print(f"extern const unsigned int g_meta_aiv_type_value_offset_c310 = {offset};")